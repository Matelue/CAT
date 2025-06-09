# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""CTC decode module

NOTE (Huahuan): currently, bs=1 is hard-coded.

Reference:
https://github.com/parlance/ctcdecode
"""

import sys

import psutil
from ..shared import tokenizer as tknz
from ..shared.tokenizer import LexiconTokenizer
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared import encoder as model_zoo
from ..shared.data import sortedScpPadCollateMT5, sortedScpPadCollateMT5G2P, MT5TestDataset
from memory_profiler import profile

import numpy as np
import matplotlib.pyplot as plt

import os
import time
import pickle
import kaldiio
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder as CTCDecoder

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

    testset = MT5TestDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollateMT5(args.hgf_tokenizer) if args.is_P2G else sortedScpPadCollateMT5G2P(args.hgf_tokenizer),
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

    results = {}
    
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len, attn_mask = batch
            batch_size = x.size(0)
            
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            
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
                                                output_logits=False,
                                                use_cache=True
                                                )
            model_out = model.model.model.generate(x, generation_config)
            scores = model_out["sequences_scores"].to("cpu").view(batch_size,-1) if args.beam_size > 1 else model_out["sequences"].to("cpu")
            hypo_ids = model_out["sequences"].view(batch_size, args.beam_size, -1)
            # token_hypos = model_out["sequences"].to("cpu").view(batch_size, args.beam_size, -1)
            
            text_hypos = [model.model.tokenizer.batch_decode(hypo_ids[j], skip_special_tokens=True) for j in range(hypo_ids.size(0))]
            # text_hypos = [model.model.tokenizer.batch_decode(token_hypos[i], skip_special_tokens=True) for i in range(batch_size)]
            # text_hypos = [' '.join(text) for text in text_hypos] if not args.is_P2G else text_hypos
            
            if args.store_ark:
                logits = model_out["logits"]
                for j in key:
                    results[j] = logits[0].cpu().numpy() 
            
            q_out.put(
                tuple(
                    (
                        key[i],
                        {
                            bid: (
                                score.tolist(),
                                '' if len(hypo) == 0 else hypo
                            )
                            for bid, (score, hypo) in enumerate(zip(scores[i], text_hypos[i]))
                        }
                    )
                    for i in range(batch_size)
                ),
                block=True,
            )

            del batch
            del model_out
            del hypo_ids
            del text_hypos
            del scores
    q_out.put(None, block=True)
    if args.store_ark:
        output_dir = os.path.join(os.path.dirname(args.output_prefix),"ark")
        os.makedirs(output_dir, exist_ok=True)
        kaldiio.save_ark(os.path.join(output_dir, f"decode.{pid+1}.ark"), results)

def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask

               
def beamsearch_score(logp_y: torch.tensor, 
               beam_size: int, 
               batch_size: int,
               eos_id: int,
               end_flag: torch.tensor,
               hyps: torch.tensor,
               scores: torch.tensor,
               device: str = 'cpu'):
    top_k_logp, top_k_index = logp_y.topk(beam_size)  # (B*N, N)
    # 掩码已结束序列
    top_k_logp = mask_finished_scores(top_k_logp, end_flag)
    top_k_index = mask_finished_preds(top_k_index, end_flag, eos_id)
    # 加入历史分数
    scores = scores + top_k_logp  # (B*N, N), broadcast add
    scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
    scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
    scores = scores.view(-1, 1)

    base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])  # (B, N)
    base_k_index = base_k_index * beam_size * beam_size
    best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)
    best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
    # best_hyps_index = best_k_index // beam_size
    best_hyps_index = torch.div(best_k_index, beam_size, rounding_mode='trunc')
    last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
    hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (B*N, i+1)
    end_flag = torch.eq(hyps[:, -1], eos_id).view(-1, 1)
    return hyps, end_flag, scores


@torch.no_grad()
def build_model(args: argparse.Namespace):
    assert (
        args.resume is not None
    ), "Trying to decode with uninitialized parameters. Add --resume"
    import importlib

    interface = importlib.import_module(args.built_model_by)  ## RAM=430MB
    model = interface.build_model(coreutils.readjson(args.config), dist=False)  ## RAM=2984MB
    # model.clean_unpickable_objs()
    # checkpoint = torch.load(args.resume, map_location="cpu")  ## RAM=10848 MB
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

def get_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 2)  # 转换为 MB
    print(f"Current RAM usage: {ram_usage:.2f} MB")

def print_gpu_utilization():
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
    gpu_utilization = torch.cuda.utilization()  # in %
    print(f"GPU Memory Allocated: {gpu_memory} MB")
    print(f"GPU Utilization: {gpu_utilization}%")

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

def plot_attention_weights(attention_weights, out_path):
    # 获取 batch_size、seqlen 和 dim
    attention_weights = attention_weights[0].cpu().detach()
    num_head, seqlen, srclen = attention_weights.shape

    # 创建图形和轴
    fig, ax = plt.subplots(nrows=num_head, ncols=1, figsize=(10, 4 * num_head))

    # 循环遍历每个样本的注意力权重
    for i in range(num_head):
        ax_i = ax[i] if num_head > 1 else ax  # 如果只有一个样本，不需要使用子图

        # 绘制热图
        cax = ax_i.matshow(attention_weights[i], cmap='viridis')

        # 添加颜色条
        fig.colorbar(cax, ax=ax_i)

        # 设置轴标签
        ax_i.set_xticks(np.arange(srclen))  # 修正这里使用 srclen 而不是 seqlen
        ax_i.set_yticks(np.arange(seqlen))
        ax_i.set_xticklabels(np.arange(srclen) + 1)  # 注意索引从0开始，这里加1以匹配实际序列
        ax_i.set_yticklabels(np.arange(seqlen) + 1)  # 注意索引从0开始，这里加1以匹配实际维度

        # 设置轴标题
        ax_i.set_xlabel('Input Sequence')
        ax_i.set_ylabel('Output Sequence')
        ax_i.set_title(f'Attention Weights - head{i + 1}')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图形到文件
    plt.savefig(out_path)

    # 显示图形（可选）
    plt.show()


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="Seq2Seq decoder.", training=False, isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="./decode")
    parser.add_argument("--lm-path", type=str, help="Path to KenLM model.")
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
        default="cat.aed.train_mt5",
        help="Tell where to import build_model() function. defautl: cat.aed.train",
    )
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
