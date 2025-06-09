# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""AED decode module

"""

import sys

import psutil
from ..shared import tokenizer as tknz
from ..shared.tokenizer import LexiconTokenizer
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared import encoder as model_zoo
from ..shared.data import sortedScpPadCollate, sortedScpPadCollateMT5G2P, ScpDataset
from memory_profiler import profile

import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
# from py3nvml import py3nvml

import os
import time
import pickle
import kenlm
import kaldiio
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder as CTCDecoder
from torch.nn import  CrossEntropyLoss

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model file: {}".format(args.tokenizer)
        )

    if args.gpu:
        world_size = torch.cuda.device_count()
        if args.nj != -1 and args.nj < world_size:
            world_size = args.nj
    else:
        if args.nj == -1:
            world_size = max(os.cpu_count() // 2, 1)
        else:
            world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    try:
        mp.set_start_method("spawn")
    except RuntimeError as re:
        print(re)

    q_data = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q_data))
    producer.start()

    q_out = mp.Queue(maxsize=1)
    consumer = mp.Process(target=datawriter, args=(args, q_out))
    consumer.start()

    if args.gpu:
        model = None
    else:
        model = build_model(args)
        model.share_memory()

    mp.spawn(worker, nprocs=world_size, args=(args, q_data, q_out, model))

    producer.join()
    consumer.join()
    del q_data
    del q_out


def dataserver(args, q: mp.Queue):

    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollate()
    )

    t_beg = time.time()
    for batch in tqdm(
        testloader, desc="AED decode", total=len(testloader), leave=False
    ):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size + 1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print(
        "Time = {:.2f} s | RTF = {:.2f} ".format(
            t_dur, t_dur * args.world_size / n_frames * 100
        )
    )
    print (n_frames)


def datawriter(args, q: mp.Queue):
    cnt_done = 0
    nbest = {}
    transcript = []

    while True:
        nbestlist = q.get(block=True)  # type: Tuple[str, Dict[int, Tuple[float, str]]]
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        for i in range(len(nbestlist)):
            key, content = nbestlist[i]
            nbest[key] = content
            transcript.append(f"{key}\t{content[0][1]}\n")
        del nbestlist

    with open(args.output_prefix, "w", encoding="utf-8") as fo:
        for l in transcript:
            fo.write(l)

    with open(args.output_prefix + ".nbest", "wb") as fo:
        pickle.dump(nbest, fo)
    
    # with open(args.output_prefix + ".nbest.txt", "wb") as fo:
    #         fo.write(key"\t"nbest[key])


@profile
def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue,
    model: model_zoo.MT5FromPretrainedModel,
):
    torch.set_num_threads(args.thread_per_woker)
    if args.gpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args).cuda(device)
    else:
        assert model is not None
        device = "cpu"

    # tokenizer = tknz.load(args.tokenizer)
    if args.lm_path is not None:
        word_tokenizer = tknz.load(args.lm_path + "/tokenizer_rmsymb_lm.tknz")
        lm = kenlm.LanguageModel(args.lm_path + "/4gram.arpa")

    results = {}
    from collections import defaultdict
    sp_time = defaultdict(float)
    
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            batch_size = x.size(0)
            
            
            x = x.to(device)
            
            # model_out = model(x, x_len, y=None, attn_mask=attn_mask)
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                                                _from_model_config=True,
                                                eos_token_id=1,
                                                pad_token_id=0,
                                                decoder_start_token_id=0,
                                                max_length=256,
                                                length_penalty=1.0,        
                                                num_beams=args.beam_size,
                                                num_return_sequences=args.beam_size,
                                                return_dict_in_generate=True,
                                                output_scores=True,
                                                output_logits=True,
                                                use_cache=True
                                                )
            B = model.sample_beam * generation_config.num_beams  # s2p_beam * p2g_beam
            
            start_time = time.time()
            
            logits_s2p_enc, logits_lens_s2p_enc = model.s2p_encoder(x, x_len)
            logits_s2p_enc = torch.log_softmax(logits_s2p_enc, dim=-1)
            logits_lens_s2p_enc = logits_lens_s2p_enc.to(torch.int32)
            sp_time["s2p_fd_time"] += time.time() - start_time

            phn_beam_results, _, _, phn_result_lens = model.phn_searcher.decode(
                logits_s2p_enc.cpu(), logits_lens_s2p_enc
            )
            
            
            
            if model.sample_beam > 0:
                # indices = torch.randperm(model.phn_searcher._beam_width)[:model.sample_beam]
                indices = torch.arange(model.sample_beam)
                phn_beam_results = phn_beam_results[:, indices, :]
                phn_result_lens = phn_result_lens[:, indices]
            
            
            N, K, T = phn_beam_results.shape
            batch_z_lens = phn_result_lens.view(N*K).to(device)
            z_seq_len = torch.max(batch_z_lens)
            phn_mask = torch.arange(z_seq_len, device=device)[None, :] < batch_z_lens[:, None].to(device)
            batch_z_results = torch.split(phn_beam_results.view(N*K,T), z_seq_len, dim=1)[0].to(device) * phn_mask
            logits_s2p_enc_beam = logits_s2p_enc.unsqueeze(1).repeat(1, K, 1, 1).reshape(N*K, T, -1) 
            logits_lens_s2p_enc_beam = logits_lens_s2p_enc.unsqueeze(1).repeat(1, K).reshape(N*K)

            s2p_loss = model.ctc_loss(logits_s2p_enc_beam.transpose(0, 1), batch_z_results, logits_lens_s2p_enc_beam, batch_z_lens).to(torch.float64)
            
            # print_gpu_utilization()
            text_hyp, text_y_ids, seq_scores = model.p2g_mt5_decode(batch_z_results, batch_z_lens, generation_config, return_y_ids = True)
            seq_scores = - seq_scores.view(N, B)

            text_hyp_beam = [text_hyp[i:i + B] for i in range(0, len(text_hyp), B)]
            text_y_ids_beam = text_y_ids.view(N, B, -1)
            del text_hyp
            del text_y_ids
            torch.cuda.empty_cache() 
            # print_gpu_utilization()
            
            uniq_y_results = []
            uniq_y_index = []
            for text_y_ids_b in text_y_ids_beam:
                uniq_y_results_b, uniq_y_index_b, y_counts = torch.unique(text_y_ids_b, dim=0, sorted=False, return_inverse=True, return_counts=True)
                uniq_y_results.append(uniq_y_results_b)
                uniq_y_index.append(uniq_y_index_b)  ## 每个y对应的z索引
            
            # uniq_y_results, uniq_y_index, y_counts = torch.unique(text_y_ids_beam, dim=1, sorted=False, return_inverse=True, return_counts=True)
            

            y_num_uniq = torch.tensor([uniq_y_results[i].shape[0] for i in range(N)])
            
            if args.lm_path is not None:
                log_lm_score = torch.tensor([lm.score(
                                ' '.join(map(str, (word_tokenizer.encode(model.p2g_model.tokenizer.decode(text_hy_seq, skip_special_tokens=True))))
                            ), bos = True, eos = True)
                            for text_hy_seq in uniq_y_results
                            ], dtype=torch.float64, device=device)
                # log_lm_score = torch.tensor([lm.score(' '.join(word_seq), bos = True, eos = True) for word_seq in y_word], dtype=torch.float64, device=device)
                log_lm_score = torch.log(torch.pow(10, log_lm_score))
            
            acc_log_p = torch.Tensor().to(device)
            y_indices = []

            y_index = [[torch.nonzero(uniq_y_index[batch] == i).squeeze(1) for i in range(y_num_uniq[batch])] for batch in range(N)]  ##每个uniq_y对应的y的索引
            # y_index = torch.stack([torch.stack(row) for row in y_index])
            z_index = [[torch.div(y_index[batch][i], generation_config.num_beams, rounding_mode='floor') for i in range(y_num_uniq[batch])] for batch in range(N)]  ##每个uniq_y对应的z的索引
            # z_index = torch.stack([torch.stack(row) for row in z_index])
            batch_text_hyp =[]
            seq_scores_b = []
            # batch_text_hyp = [text_hyp_beam[batch][i] for batch in range(N) for idx in y_index[batch] for i in idx]  ##每个uniq_y对应的text
            for batch in range(N):
                for idx in y_index[batch]:
                    for i in idx:
                        batch_text_hyp.append(text_hyp_beam[batch][i])
                        seq_scores_b.append(seq_scores[batch][i].item())
            
            del seq_scores
            torch.cuda.empty_cache()

            seq_scores_b = torch.Tensor(seq_scores_b).view(N,B).to(device)
            batch_z_results = batch_z_results.view(N, K, -1)
            batch_z_lens = batch_z_lens.view(N, K)
            # batch_z_lens = batch_z_lens.view(N, K).unsqueeze(1).repeat(1, int(B/K), 1).view(N * B) 
            
            # batch_z_results_b = batch_z_results[torch.arange(N), z_index]

            batch_z_results_b = torch.Tensor().to(device, dtype=batch_z_results.dtype)
            batch_z_lens_b = torch.Tensor().to(device, dtype=batch_z_lens.dtype)
            s2p_loss_b = torch.Tensor().to(device, dtype=s2p_loss.dtype)
            for batch in range(N):
                batch_z_results_b_ = torch.Tensor().to(device, dtype=batch_z_results.dtype)
                batch_z_lens_b_ = torch.Tensor().to(device, dtype=batch_z_lens.dtype)
                s2p_loss_b_ = torch.Tensor().to(device, dtype=s2p_loss.dtype)
                for idx in range(len(z_index[batch])):
                    batch_z_results_idx = batch_z_results[batch][z_index[batch][idx]]
                    batch_z_lens_idx = batch_z_lens[batch][z_index[batch][idx]]
                    s2p_loss_idx = s2p_loss.view(N,K)[batch][z_index[batch][idx]]
                    batch_z_results_b_ = torch.cat((batch_z_results_b_, batch_z_results_idx), dim=0)
                    batch_z_lens_b_ = torch.cat((batch_z_lens_b_, batch_z_lens_idx), dim=0)
                    s2p_loss_b_ = torch.cat((s2p_loss_b_, s2p_loss_idx), dim=0)
                batch_z_results_b = torch.cat((batch_z_results_b, batch_z_results_b_.unsqueeze(0)), dim=0)
                batch_z_lens_b = torch.cat((batch_z_lens_b, batch_z_lens_b_.unsqueeze(0)), dim=0)
                s2p_loss_b = torch.cat((s2p_loss_b, s2p_loss_b_.unsqueeze(0)), dim=0)
            batch_z_results_b = batch_z_results_b.view(N * B, -1)
            batch_z_lens_b = batch_z_lens_b.view(N * B)

            
            # p2g_loss, input_lb_lens = model.p2g_mt5_forward(batch_z_results_b, batch_z_lens_b, batch_text_hyp)

            # p2g_loss_seq = torch.sum(p2g_loss, dim=-1).view(N, B)
            z_lens = [[len(sublist) for sublist in z_index[batch]] for batch in range(N)]
            seq_scores_seq_b = [seq_scores_b[i].split(z_lens[i]) for i in range(N)]
            # p2g_loss_seq_b = [p2g_loss_seq[i].split(z_lens[i]) for i in range(N)]
            s2p_loss_b = [s2p_loss_b[i].split(z_lens[i]) for i in range(N)]

            indices = []
            # sp_time["p2g_rag_time"] += time.time() - start_time
            for batch in range(N):
                # len_b = len(p2g_loss_seq_b[batch])
                len_b = len(seq_scores_seq_b[batch])
                log_p_b = torch.empty(len_b, device=device, dtype=seq_scores_b.dtype)
                for i in range(len_b):
                    # log_p = torch.logsumexp(-s2p_loss_b[batch][i] - p2g_loss_seq_b[batch][i], 0, keepdim=True)
                    log_p = torch.logsumexp(-s2p_loss_b[batch][i] - seq_scores_seq_b[batch][i], 0, keepdim=True)
                    log_p_b[i] = log_p
                indices_b = torch.sort(log_p_b, descending=True)
                indices.append(indices_b)
            
            # log_p = torch.logsumexp(-s2p_loss_b - p2g_loss_seq, 1, keepdim=True)
            # _, indices = torch.sort(acc_log_p, descending=True)
            
            
            q_out.put(
                tuple(
                    (
                        key[i],
                        {
                            bid: (
                                score.tolist(), model.p2g_model.tokenizer.decode(hypo, skip_special_tokens=True)
                            )
                            for bid, (score, hypo) in enumerate(zip(indices[i][0], uniq_y_results[i][indices[i][1]]))
                        }
                    )
                    for i in range(N)  # N: batch size
                ),
                block=True,
            )
            
            
    q_out.put(None, block=True)
    if args.store_ark:
        output_dir = os.path.join(os.path.dirname(args.output_prefix),"ark")
        os.makedirs(output_dir, exist_ok=True)
        kaldiio.save_ark(os.path.join(output_dir, f"decode.{pid+1}.ark"), results)


@torch.no_grad()
def build_model(args: argparse.Namespace):
    assert (
        args.resume is not None
    ), "Trying to decode with uninitialized parameters. Add --resume"
    import importlib

    interface = importlib.import_module(args.built_model_by)
    model = interface.build_model(coreutils.readjson(args.config), dist=False)
    # model.clean_unpickable_objs()
    # checkpoint = torch.load(args.resume, map_location="cpu")
    checkpoint = torch.load(args.resume, map_location=torch.device("cuda"))
    model = coreutils.load_checkpoint(model, checkpoint)
    model.eval()
    return model


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def generate_attention_mask(lens_matrix):
        seq_len = torch.max(lens_matrix)
        attention_mask = torch.arange(seq_len, device=lens_matrix.device)[
            None, :
        ] < lens_matrix[:, None].to(lens_matrix.device)
        # attention_mask = (1.0 - attention_mask.to(self.dtype)) * torch.finfo(self.dtype).min
        return attention_mask.to(torch.float32)

def get_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 2)  # 转换为 MB
    print(f"Current RAM usage: {ram_usage:.2f} MB")

def print_gpu_utilization():
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
    gpu_utilization = torch.cuda.utilization()  # in %
    print(f"GPU Memory Allocated: {gpu_memory} MB")
    print(f"GPU Utilization: {gpu_utilization}%")
# def get_gpu_usage():
#     py3nvml.nvmlInit()

#     # 获取 GPU 0 和 GPU 1 的内存信息
#     handle0 = py3nvml.nvmlDeviceGetHandleByIndex(0)
#     handle1 = py3nvml.nvmlDeviceGetHandleByIndex(1)

#     mem_info0 = py3nvml.nvmlDeviceGetMemoryInfo(handle0)
#     mem_info1 = py3nvml.nvmlDeviceGetMemoryInfo(handle1)

#     print(f"GPU 0 memory used: {mem_info0.used / (1024 ** 2):.2f} MB")
#     print(f"GPU 1 memory used: {mem_info1.used / (1024 ** 2):.2f} MB")

#     py3nvml.nvmlShutdown()
    
def _prepare_attention_mask_for_generation(
    self,
    inputs: torch.Tensor,
    pad_token_id: Optional[int],
    eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
        return inputs.ne(pad_token_id).long()
    else:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="Seq2Seq decoder.", training=False, isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="./decode")
    parser.add_argument("--lm-path", type=str, default=None, help="Path to KenLM model.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="The 'alpha' value for LM integration, a.k.a. the LM weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.",
    )
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument(
        "--do-normalize",
        action="store_true",
        default=False,
        help="Do the log-softmax normalization before beam search.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU to do inference. Default: False.",
    )
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument(
        "--built-model-by",
        type=str,
        default="cat.aed.train_mt5_rag_batch_sample",
        help="Tell where to import build_model() function. defautl: cat.aed.train",
    )
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
