# Copyright 2024 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

"""Top interface of RAG Sequence training.
    S2P: CTC based speech to phone model.
    P2G: AED based phone to BPE model.

"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

import re
import os
import time
from ..shared.manager import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import S2P2GSpeechDataset, sortedPadCollateS2P2G
from ..shared.tokenizer import load
from transformers import GenerationConfig
# from dynamic_tanh import convert_ln_to_dyt

import argparse
import Levenshtein
from typing import *
import ctc_align
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import jiwer

def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace, **mkwargs):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if "T_dataset" not in mkwargs:
        mkwargs["T_dataset"] = S2P2GSpeechDataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = sortedPadCollateS2P2G(flatten_target=False)

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if (
        "func_eval" not in mkwargs
        and hasattr(args, "eval_error_rate")
        and args.eval_error_rate
    ):
        mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    if args.ld is None:
        tr_dataset = manager.trainloader.dl.dataset
        coreutils.distprint(
            f"  total {tr_dataset.__len__()} utterances are used in training.", args.gpu
        )

    # training
    manager.run(args)

class AMTrainer(nn.Module):
    def __init__(
        self,
        s2p_encoder: model_zoo.AbsEncoder,
        phn_decoder: CTCBeamDecoder,
        p2g_model: model_zoo.MT5FromPretrainedModel,
        is_rag_tok: bool = False,
        use_beam_search: bool = False,
        sample_beam: int = -1,
        sample_size: int = -1,
        num_samples: int = 1,
        fold_size : int = 1,
        T_weight_s2p: float = 1.0,
        s2p_tknz: str = None
    ):
        super().__init__()

        self.s2p_encoder = s2p_encoder
        self.phn_searcher = phn_decoder
        self.p2g_model = p2g_model
        self.is_rag_tok = is_rag_tok
        self.use_beam_search = use_beam_search
        self.sample_beam = sample_beam
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.fold_size = fold_size
        self.T_weight_s2p = T_weight_s2p
        # assert self.sample_beam <= self.phn_searcher._beam_width, f"number of sample must be less than beam width."
        
        self.dtype = torch.float32
            
        self.ctc_loss = nn.CTCLoss(reduction='none',zero_infinity=True)
        # self.bpe_tokenizer = load("/home/saier/CAT/egs/jsa/data/dict_pl/tokenizer_bpe500.tknz")
        assert s2p_tknz is not None, f"S2P phoneme tokenizer is required for phoneme decoding."
        self.phn_tokenizer = load(s2p_tknz)


    # def forward(self, x, lx, y, ly, uids, y_pid, ly_pid, y_char, ly_char, nbest_list=None, nbest_lens=None):
    def forward(self, x, lx, z, lz, y, ly):
        # s2p_encoder forward
        logits_s2p_enc, logits_lens_s2p_enc = self.s2p_encoder(x, lx)
        logits_s2p_enc = logits_s2p_enc / self.T_weight_s2p
        logits_s2p_enc = torch.log_softmax(logits_s2p_enc, dim=-1)
        
        batch_size = lx.shape[0]
        ly = ly.to(torch.int)
        lz = lz.to(torch.int)

        # S2P ctc beamSearch decoding
        if self.use_beam_search:
            beam_results, _, _, beam_result_lens = self.phn_searcher.decode(logits_s2p_enc, logits_lens_s2p_enc)
            
        # S2P sample decoding
        if self.sample_size > 0:
            samples_results, samples_result_lens = self._sample(logits_s2p_enc.detach().exp(), logits_lens_s2p_enc, n_samples=self.sample_size)
        
            pad_mask = torch.arange(samples_results.size(2)).expand(samples_results.size(0), samples_results.size(1), samples_results.size(2)).cuda() < samples_result_lens.unsqueeze(-1)
            samples_results = samples_results * pad_mask
            samples_results[~pad_mask] = 0

            # samples_results, samples_result_lens = self.uniq_sample(samples_results, samples_result_lens)
        if self.use_beam_search and self.sample_size > 0:
            beam_sample_results = torch.cat((beam_results.cuda(), samples_results.transpose(0,1)), dim=1)  # batch_size x (beam_size+sample_size) x seq_len
            beam_sample_result_lens = torch.cat((beam_result_lens.cuda(), samples_result_lens.transpose(0,1)), dim=1)
            B = self.phn_searcher._beam_width + self.sample_size
        elif self.use_beam_search:
            beam_sample_results = beam_results
            beam_sample_result_lens = beam_result_lens
            B = self.phn_searcher._beam_width
        else:
            beam_sample_results = samples_results.transpose(0,1)
            beam_sample_result_lens = samples_result_lens.transpose(0,1)
            B = self.sample_size
        
        # random select n beam from beam_results,            
        if self.sample_beam > 0:
            selected_results = []
            selected_lengths = []
            for _ in range(self.num_samples):
                indices = torch.randperm(B)[:self.sample_beam]
                sampled_results = beam_sample_results[:, indices, :]
                sampled_lengths = beam_sample_result_lens[:, indices]

                selected_results.append(sampled_results)
                selected_lengths.append(sampled_lengths)
                
        loss = torch.zeros(1, 1).cuda()
        for s in range(self.num_samples):
            beam_sample_results_ = selected_results[s].view(selected_results[s].size(0), self.fold_size, int(self.sample_beam/self.fold_size), selected_results[s].size(2))
            beam_sample_result_lens_ = selected_lengths[s].view(selected_lengths[s].size(0), self.fold_size, int(self.sample_beam/self.fold_size))
            p2g_loss_fold = []
            s2p_loss_fold = []
            for i in range(self.fold_size):
                beam_sample_results = beam_sample_results_[:, i, :, :]
                beam_sample_result_lens = beam_sample_result_lens_[:, i, :]


                N, K, T = beam_sample_results.shape
                p_list_tok = torch.Tensor().cuda()
                p_list_seq = torch.Tensor().cuda()

                pad_mask = torch.arange(T).expand(N, K, T).cuda() < beam_sample_result_lens.cuda().unsqueeze(-1)
                z_new = beam_sample_results.cuda() * pad_mask
                z_new[~pad_mask] = 0
                z_new_in_batch = z_new.reshape(N*K, -1)
                lens_in_batch = beam_sample_result_lens.reshape(N*K)
                    
                logits_s2p_enc_beam = logits_s2p_enc.unsqueeze(1).repeat(1, K, 1, 1).reshape(N*K, T, -1)  # batch_size * beam_size x seq_len x numclass
                logits_lens_s2p_enc_beam = logits_lens_s2p_enc.unsqueeze(1).repeat(1, K).reshape(N*K)
                s2p_loss = self.ctc_loss(logits_s2p_enc_beam.transpose(0, 1), z_new_in_batch.cuda(), logits_lens_s2p_enc_beam.to(torch.int), lens_in_batch.to(torch.int).cuda())  #  batch_size * beam_size x 1
                s2p_loss = s2p_loss.view(N, K)
                s2p_loss_fold.append(s2p_loss)

                y_beam= [item for item in y for i in range(K)]
                ly_beam = ly.unsqueeze(1).repeat(1,K).reshape(N*K)
                p2g_loss, input_y_lens = self.p2g_mt5_forward(z_new_in_batch.cuda(), lens_in_batch.cuda(), y_beam, ly_beam)  # beam_size * batch_size x seq_len
                p2g_loss = p2g_loss.view(N, K, -1)  # batch_size x beam_size x seq_len
                input_y_lens = input_y_lens.view(N, K)
                p2g_loss_fold.append(p2g_loss)
                
            s2p_loss = torch.cat(s2p_loss_fold, dim=1)
            p2g_loss = torch.cat(p2g_loss_fold, dim=1)
            if self.is_rag_tok:
                p_list_tok = s2p_loss[:,:,None] + p2g_loss  # batch_size x beam_size x seq_len 
            else:
                p2g_loss_seq = torch.sum(p2g_loss, dim=-1)  # batch_size x beam_size  
                p_list_seq = - (s2p_loss + p2g_loss_seq)  # batch_size x beam_size
                        
            if self.is_rag_tok:
                loss = torch.logsumexp(p_list_tok, 1)  # batch_size x seq_len
                total_loss = torch.sum(loss, dim=-1)  # batch_size x 1
            else:
                total_loss = - torch.logsumexp(p_list_seq.view(N,self.sample_beam), 1)  # batch_size x 1
                
            loss = loss + torch.mean(total_loss)
        
        return loss

    
    def validate_zlen_and_pad(self, zlists, zlens):
        zlist_new = zlists.copy()
        zlens_new = zlens.clone()
        if (zlens_new == 0).any():
            # (num_utt % batch_size) item not covered in self.zlist because of mini-batch sampling
            index = torch.nonzero(zlens_new == 0).squeeze(dim=1)
            for i in index:
                zlens_new[i] = 1
                zlist_new[i] = torch.ones([1], dtype=torch.int32)

        return pad_sequence(zlist_new, batch_first=True, padding_value=0), zlens_new
    
    def clean_unpickable_objs(self):
        pass

    def _get_wer(
        self, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor, type = "WER"
    ):
        acc_err = 0.
        acc_cnt = 0
        snt_per = []
        for x, xlen, y, ylen in zip(xs, lx, ys, ly):
            if type == "PER":
                x1 = [x[:xlen].tolist()]
                y1 = [y[:ylen].tolist()]
                err, cnt = cal_wer(y1, x1)
            elif type == "WER":
                x1 = [self.bpe_tokenizer.decode(x[:xlen].tolist())]
                y1 = [self.bpe_tokenizer.decode(y[:ylen].tolist())]
                measure = jiwer.compute_measures(y1, x1)
                cnt = measure['hits'] + measure['substitutions'] + measure['deletions']
                err = measure['substitutions'] + measure['deletions'] + measure['insertions']
            else:
                raise TypeError(f"type {type} is illegal!")
            acc_err += err
            acc_cnt += cnt
            snt_per.append(err / cnt)
        return torch.tensor([acc_err / acc_cnt], device="cpu", dtype=torch.float32), torch.tensor(snt_per, device="cpu", dtype=torch.float32)
    
    def get_wer(
        self, xs: torch.Tensor, lx: torch.Tensor, zs: torch.Tensor, lz: torch.Tensor, ys: torch.Tensor, ly: torch.Tensor
    ):
        # if self.attach["decoder"] is None:
        #     raise RuntimeError(
        #         f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
        #     )

        bs = xs.size(0)
        logits, lx = self.s2p_encoder(xs, lx)

        # y_samples: (N, k, L), ly_samples: (N, k)
        if self.use_beam_search:
            phn_samples, _, _, lphn_samples = self.phn_searcher.decode(
                logits.float().cpu(), lx.cpu()
            )
            lphn_samples = lphn_samples[:,0]
            phn_hypos = [phn_samples[n, 0, : lphn_samples[0]].tolist() for n in range(bs)]
        else:
            samples_results, samples_result_lens = self._sample(logits.detach().exp(), lx, n_samples=1)
        
            pad_mask = torch.arange(samples_results.size(2)).expand(samples_results.size(0), samples_results.size(1), samples_results.size(2)).cuda() < samples_result_lens.unsqueeze(-1)
            samples_results = samples_results * pad_mask
            samples_results[~pad_mask] = 0
            phn_samples = samples_results.transpose(0,1)
            lphn_samples = samples_result_lens.transpose(0,1)
            phn_hypos = [phn_samples[n, 0, : lphn_samples[n, 0]].tolist() for n in range(bs)]

        
        
        model_out, y_ids = self.p2g_mt5_forward(phn_hypos, lphn_samples, ys, ly, return_y_ids=True, return_dict = True)
        logits = torch.log_softmax(model_out[1], dim = -1, dtype=torch.float32)
        hypos = torch.argmax(logits, dim = -1).tolist()
        gt = y_ids.tolist()

        return cal_wer(gt, hypos)
    
    def p2g_mt5_forward(self, input_x, lens_input_x, input_y, lens_input_y: Optional[Union[torch.Tensor, List[int]]] = None, T_weight: float = 1.0, return_y_ids: bool = False, return_dict: bool = False):
        if not isinstance(input_x, torch.Tensor):
            input_x = [input_x[i][:lens_input_x[i]] for i in range(len(input_x))]
        else:
            input_x = [input_x[i][:lens_input_x[i]].tolist() for i in range(input_x.size(0))]
        if not isinstance(input_y, torch.Tensor):
            if lens_input_y is not None:
                input_y = [input_y[i][:lens_input_y[i]] for i in range(len(input_y))]
        else:
            input_y = [input_y[i][:lens_input_y[i]].tolist() for i in range(input_y.size(0))]
        input_x = self.phn_tokenizer.decode(input_x)
        input_x = [''.join(input_x[i].split()) for i in range(len(input_x))]
        # input_y = self.bpe_tokenizer.decode(input_y) 
        input_p2g = self.p2g_model.tokenizer(input_x, text_target=input_y, max_length=250, padding='longest', return_tensors='pt')
        input_lb_lens = (input_p2g["labels"] != 0).sum(dim=1)
        # print(f"Before forward pass: {torch.cuda.memory_allocated()} bytes")
        p2g_loss_new = self.p2g_model(input_p2g["input_ids"].cuda(), input_p2g["labels"].cuda(), attn_mask = input_p2g["attention_mask"].cuda(), return_dict = return_dict, loss_reduction = "none", T_weight = T_weight)
        # plot_dis(p2g_loss_new[1])
        if return_y_ids:
            return p2g_loss_new, input_p2g["labels"]
        else:
            return p2g_loss_new, input_lb_lens
    
    def p2g_mt5_decode(self, input_x, lens_input_x, config, return_y_ids: bool = False):
        p2g_ge_time = 0
        input_x = [input_x[i][:lens_input_x[i]].tolist() for i in range(input_x.size(0))]
        input_x = self.phn_tokenizer.decode(input_x)
        input_x = [''.join(input_x[i].split()) for i in range(len(input_x))]
        start_time = time.time()
        input_p2g = self.p2g_model.tokenizer(input_x, max_length=250, padding='longest', return_tensors='pt')
        p2g_dec_out = self.p2g_model.model.generate(input_p2g['input_ids'].cuda(), config)  ## mt5 decode
        p2g_ge_time += time.time() - start_time
        if config.return_dict_in_generate:
            text_hypos = self.p2g_model.tokenizer.batch_decode(p2g_dec_out["sequences"], skip_special_tokens=True)  # 
            if config.num_beams > 1:
                seq_scores = p2g_dec_out["sequences_scores"]
            else:
                scores = p2g_dec_out["logits"]
                token_ids = p2g_dec_out["sequences"][:, 1:len(scores)+1]
                step_scores = torch.stack(scores, dim=1)
                log_probs = torch.nn.functional.log_softmax(step_scores, dim=-1)
                log_probs = log_probs[:, :-1, :]
                token_log_probs = torch.gather(log_probs, dim=2, index=token_ids[:, :log_probs.size(1)].unsqueeze(-1)).squeeze(-1)
                pad_token_id = config.pad_token_id
                mask = (token_ids[:, :log_probs.size(1)] != pad_token_id).float()
                token_log_probs = token_log_probs * mask
                seq_scores = token_log_probs.sum(dim=-1)
            if return_y_ids:
                return text_hypos, p2g_dec_out["sequences"], seq_scores
            else:
                return text_hypos, seq_scores
        else:
            text_hypos = self.p2g_model.tokenizer.batch_decode(p2g_dec_out, skip_special_tokens=True)
            return text_hypos
        
    def _sample(self, probs, lx, n_samples=None):
        N, T, V = probs.shape
        K = n_samples
        # (NT, K)
        samples = torch.multinomial(probs.view(-1, V), K, replacement=True).view(
            N, T, K
        )
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ys, ly = ctc_align.align_(
            samples.transpose(1, 2).contiguous().view(-1, T),
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1),
        )
        return ys.view(N, K, T).transpose(0,1), ly.view(N, K).transpose(0,1)
    
    def uniq_sample(self, samples, sample_lens):
        samples_list = []
        samples_lens_list = []
        K, N, T = samples.shape
        max_sample_len = max(sample_lens.transpose(0,1).reshape(-1)) if max(sample_lens.transpose(0,1).reshape(-1)) !=0 else 1
        
        pad_mask = torch.arange(T).expand(K, N, T).cuda() < sample_lens.unsqueeze(-1)
        sample_pad = samples * pad_mask
        sample_pad[~pad_mask] = 0
        
        samples_list = torch.Tensor().to(samples.device, dtype=samples.dtype)
        samples_lens_list = torch.Tensor().to(sample_lens.device, dtype=sample_lens.dtype)
        beam=0
        for results, lens in zip(samples, sample_lens): 
            z_sample = [results[batch][:lens[batch]] for batch in range(N)]
            z_sample_in_batch, zlens_sample_in_batch = validate_zlen_and_pad(z_sample, lens)
            samples_list = [torch.cat((samples_list if beam == 0 else samples_list[i], z_sample_in_batch[i].unsqueeze(0)), dim=0) for i in range(N)]
            samples_lens_list = [torch.cat((samples_lens_list if beam == 0 else samples_lens_list[i], zlens_sample_in_batch[i].unsqueeze(0)), dim=0) for i in range(N)]
            beam += 1


        samples_tensor = pad_sequence(samples_list, batch_first=True, padding_value=0).transpose(1, -1).view(-1, max_sample_len).transpose(0,1)
        samples_lens_tensor = torch.tensor(samples_lens_list, dtype=torch.int).transpose(0,1)
    
        unique_samples = [torch.unique(samples_tensor[i], dim=0) for i in range(N)]
        unique_sample_lens = [torch.unique(samples_lens_tensor[i], dim=0) for i in range(N)]

        beam_lengths = torch.tensor([ub.size(0) for ub in unique_samples])
        max_uniq_beam = beam_lengths.max().item()
        
        unique_samples_batch = torch.stack([pad_or_truncate(ub, max_uniq_beam) for ub in unique_samples])
        unique_sample_lens_batch = torch.stack([pad_or_truncate(ub, max_uniq_beam) for ub in unique_sample_lens])
        
        return unique_samples_batch, unique_sample_lens_batch
        
def pad_or_truncate(ub, max_beam):
    repeats = max_beam // ub.size(0)
    remainder = max_beam % ub.size(0)
    return ub.repeat(repeats + 1, 1)[:max_beam]

def validate_zlen_and_pad(zlists, zlens):
    zlist_new = zlists.copy()
    zlens_new = zlens.clone()
    if (zlens_new == 0).any():
        # (num_utt % batch_size) item not covered in self.zlist because of mini-batch sampling
        index = torch.nonzero(zlens_new == 0).squeeze(dim=1)
        for i in index:
            zlens_new[i] = 1
            zlist_new[i] = torch.ones([1], dtype=torch.int32, device=zlens.device)

    return zlist_new, zlens_new
    
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_comparison(logits, sample_seq, beam_seq, f_save):
    """
    画图对比采样、束搜索和Softmax输出的概率分布，并保存为图片。
    
    参数:
    logits: ndarray, 维度为 (seq_len, classes)，Softmax前的logits输出
    sample_seq: list or ndarray, 维度为 (seq_len)，采样生成的序列
    beam_seq: list or ndarray, 维度为 (seq_len)，束搜索生成的序列
    f_save: str, 保存图像的路径
    
    返回:
    None
    """
    seq_len, classes = logits.shape

    # 计算Softmax输出的概率分布
    softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # 创建绘图
    fig, axes = plt.subplots(seq_len, 1, figsize=(10, 15))

    for t in range(seq_len):
        ax = axes[t]
        
        # Softmax输出的概率分布
        ax.bar(range(classes), softmax_probs[t], alpha=0.5, label="Softmax", color='gray')

        # 采样选择的token的概率
        ax.scatter(sample_seq[t], softmax_probs[t, sample_seq[t]], color='blue', label="采样", zorder=5)

        # 束搜索选择的token的概率
        ax.scatter(beam_seq[t], softmax_probs[t, beam_seq[t]], color='red', label="束搜索", zorder=5)

        ax.set_title(f"时间步 {t+1}")
        ax.set_xlabel("Token")
        ax.set_ylabel("概率")
        ax.legend()

    plt.tight_layout()

    # 保存图像
    plt.savefig(f_save)
    plt.close()

    print(f"图像已保存到 {f_save}")
    

    # 绘制条形图
    
    
def cal_wer(gt: List[List[int]], hy: List[List[int]]) -> Tuple[int, int]:
    """compute error count for list of tokens"""
    assert len(gt) == len(hy)
    err = 0
    cnt = 0
    for i in range(len(gt)):
        err += Levenshtein.distance(
            "".join(chr(n) for n in hy[i]), "".join(chr(n) for n in gt[i])
        )
        cnt += len(gt[i])
    return (err, cnt)

@torch.no_grad()
def custom_evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:
    model = manager.model
    cnt_tokens = 0
    cnt_err = 0
    n_proc = dist.get_world_size()

    for minibatch in tqdm(
        testloader,
        desc=f"Epoch: {manager.epoch} | eval",
        unit="batch",
        disable=(args.gpu != 0),
        leave=False,
    ):
        feats, ilens, phn_labels, phn_lens, y_labels, y_lens = minibatch[:6]
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        phn_labels = phn_labels.cuda(args.gpu, non_blocking=True)
        phn_lens = phn_lens.cuda(args.gpu, non_blocking=True)
        y_lens = y_lens.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, ilens, phn_labels, phn_lens, y_labels, y_lens)
        cnt_err += part_cnt_err
        cnt_tokens += part_cnt_sum

    gather_obj = [None for _ in range(n_proc)]
    dist.gather_object(
        (cnt_err, cnt_tokens), gather_obj if args.rank == 0 else None, dst=0
    )
    dist.barrier()
    if args.rank == 0:
        l_err, l_sum = list(zip(*gather_obj))
        wer = sum(l_err) / sum(l_sum)
        manager.writer.add_scalar("loss/dev-token-error-rate", wer, manager.step)
        scatter_list = [wer]
    else:
        scatter_list = [None]

    dist.broadcast_object_list(scatter_list, src=0)
    return scatter_list[0]


def build_beamdecoder(cfg: dict) -> CTCBeamDecoder:
    """
    beam_size:
    num_classes:
    kenlm:
    alpha:
    beta:
    ...
    """

    assert "num_classes" in cfg, "number of vocab size is required."

    if "kenlm" in cfg:
        labels = [str(i) for i in range(cfg["num_classes"])]
        labels[0] = "<s>"
        labels[1] = "<unk>"
    else:
        labels = [""] * cfg["num_classes"]

    return CTCBeamDecoder(
        labels=labels,
        model_path=cfg.get("kenlm", None),
        beam_width=cfg.get("beam_size", 16),
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 0.0),
        num_processes=cfg.get("num_processes", 6),
        log_probs_input=True,
        is_token_based=("kenlm" in cfg),
    )

def build_model(
    cfg: dict,
    args: Optional[Union[argparse.Namespace, dict]] = None,
    dist: bool = True,
    wrapper: bool = True,
) -> Union[nn.parallel.DistributedDataParallel, AMTrainer, model_zoo.AbsEncoder]:
    """
    for ctc-crf training, you need to add extra settings in
    cfg:
        trainer:
            use_crf: true/false,
            lamb: 0.01,
            den_lm: xxx

            decoder:
                beam_size:
                num_classes:
                kenlm:
                alpha:
                beta:
                ...
        ...
    """
    if "trainer" not in cfg:
        cfg["trainer"] = {}
    

    assert "s2p_encoder" in cfg
    s2p_netconfigs = cfg["s2p_encoder"]
    s2p_net_kwargs = s2p_netconfigs["kwargs"]  # type:dict

    n_classes = s2p_net_kwargs.pop("n_classes")
    s2p_encoder = getattr(model_zoo, s2p_netconfigs["type"])(
        num_classes = n_classes, **s2p_net_kwargs
    )  # type: model_zoo.AbsEncoder
    

    assert "p2g_model" in cfg
    p2g_enc_configs = cfg["p2g_model"]
    p2g_use_dyt = p2g_enc_configs["use_dyt"] if "use_dyt" in p2g_enc_configs else False
    p2g_enc_kwargs = p2g_enc_configs["kwargs"]  # type:dict
    p2g_model = getattr(model_zoo, p2g_enc_configs["type"])(**p2g_enc_kwargs)  # type: model_zoo.AbsEncoder
    
    assert "beamDecoder" in cfg
    beamDecoder_cfg = cfg["beamDecoder"]
    phn_searcher = eval(beamDecoder_cfg["type"])(
        [""] * beamDecoder_cfg["n_classes"],
        beam_width=beamDecoder_cfg["beam_width"],
        log_probs_input=beamDecoder_cfg["log_probs_input"],
        num_processes=beamDecoder_cfg["num_processes"]
    )
    
    if s2p_netconfigs.get("freeze", False):
        s2p_encoder.requires_grad_(False)
    if p2g_enc_configs.get("freeze", False):
        p2g_model.requires_grad_(False)

    model = AMTrainer(s2p_encoder,
                      phn_searcher,
                      p2g_model,
                      **cfg["trainer"]
                      )
    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"unsupport type of args: {type(args)}")
    
    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args["gpu"])
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[args["gpu"]])

    init_checkpoint = OrderedDict()
    if "init_model" in s2p_netconfigs:
        coreutils.distprint(f"> initialize s2p_encoder from: {s2p_netconfigs['init_model']}", args["gpu"])
        s2p_enc_checkpoint = torch.load(
            s2p_netconfigs["init_model"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint, "encoder", "s2p_encoder")
    else:
        s2p_enc_checkpoint = s2p_encoder.state_dict()
        s2p_enc_checkpoint = translate_checkpoint(s2p_enc_checkpoint, "encoder", "module.s2p_encoder.model")
    init_checkpoint.update(s2p_enc_checkpoint)
    del s2p_enc_checkpoint

    if "init_model" in p2g_enc_configs:
        coreutils.distprint(f"> initialize p2g model from: {p2g_enc_configs['init_model']}", args["gpu"])
        p2g_checkpoint = torch.load(
            p2g_enc_configs["init_model"], 
            map_location=f"cuda:{args['gpu']}"
        )["model"]  # type: OrderedDict
        p2g_checkpoint = translate_checkpoint(p2g_checkpoint, "model", "p2g_model")
    else:
        p2g_checkpoint = translate_checkpoint(p2g_model.state_dict(), "model", "module.p2g_model.model")
    init_checkpoint.update(p2g_checkpoint)
    del p2g_checkpoint

    if len(init_checkpoint) != 0:
        model.load_state_dict(init_checkpoint)
        del init_checkpoint

    if p2g_use_dyt:
        model = convert_ln_to_dyt(model)
        # model = model.to("cuda")

    return model

# FIXME: following codes will be removed soon or later
########## COMPATIBLE ###########
# fmt: off
def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str) -> OrderedDict:
    """Translate checkpoint of previous version of RNN-T so that it could be loaded with the new one."""
    old_string = old_string + '.'
    new_string = new_string + '.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if old_string in k:
            k = k.replace(old_string, new_string, 1)
            new_state_dict[k] = v
    return new_state_dict
# fmt: on
#################################

from timm.layers import LayerNorm2d
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value).cuda()
        self.weight = nn.Parameter(torch.ones(normalized_shape)).cuda()
        self.bias = nn.Parameter(torch.zeros(normalized_shape)).cuda()

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


def _parser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of CTC loss (default). "
        "If specified, you should setup 'decoder' in 'trainer' configuration.",
    )
    return parser


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)


if __name__ == "__main__":
    print(
        "NOTE:\n"
        "    since we import the build_model() function in cat.ctc,\n"
        "    we should avoid calling `python -m cat.ctc.train`, instead\n"
        "    running `python -m cat.ctc`"
    )
