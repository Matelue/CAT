# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""CTC decode module

NOTE (Huahuan): currently, bs=1 is hard-coded.

Reference:
https://github.com/parlance/ctcdecode
"""

import sys
from ..shared import tokenizer as tknz
from ..shared.tokenizer import LexiconTokenizer
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.data import ScpDataset, sortedScpPadCollate, P2GTestDataset


import os
import time
import pickle
import kaldiio
import ctc_align
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder as CTCDecoder

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


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
    if args.input_scp.endswith("_phn"):
        testset = P2GTestDataset(args.input_scp)
    else:
        testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollate(),
    )

    t_beg = time.time()
    for batch in tqdm(
        testloader, desc="CTC decode", total=len(testloader), leave=False
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
        key, content = nbestlist
        nbest[key] = content
        transcript.append(f"{key}\t{content[0][1]}\n")
        del nbestlist

    with open(args.output_prefix, "w") as fo:
        for l in transcript:
            fo.write(l)

    with open(args.output_prefix + ".nbest", "wb") as fo:
        pickle.dump(nbest, fo)


def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue,
    model: AbsEncoder,
):
    torch.set_num_threads(args.thread_per_woker)
    if args.gpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args).cuda(device)
    else:
        assert model is not None
        device = "cpu"

    tokenizer = tknz.load(args.tokenizer)

    if args.lm_path is None:
        # w/o LM, labels won't be used in decoding.
        labels = [""] * tokenizer.vocab_size
        searcher = CTCDecoder(
            labels,
            beam_width=args.beam_size,
            log_probs_input=True,
            num_processes=args.thread_per_woker,
        )
    else:
        assert os.path.isfile(
            args.lm_path
        ), f"--lm-path={args.lm_path} is not a valid file."

        # NOTE: ctc decoding with an ext. LM assumes <s> -> 0 and <unk> -> 1
        labels = ["<s>", "<unk>"] + [str(i) for i in range(2, tokenizer.vocab_size)]
        searcher = CTCDecoder(
            labels,
            model_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            beam_width=args.beam_size,
            num_processes=args.thread_per_woker,
            log_probs_input=True,
            is_token_based=True,
        )
    results = {}
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            x = x.to(device)
            key = key[0]
            if args.streaming:
                logits, olens = model.chunk_infer(x, x_len)
            else:
                logits, olens = model.encoder(x, x_len)
            # results[key] = logits[0].cpu().numpy()
            if args.beam_size is not None:
                samples, samples_lens = _sample(logits.detach().exp(), olens, n_samples=args.beam_size)
                
                max_batch_size = samples.size(1)
                max_beam_size = samples.size(0)
                max_sample_len = max(samples_lens) if max(samples_lens) !=0 else 1
                samples_list = []
                samples_lens_list = []


                for results, lens in zip(samples, samples_lens): 
                    z_sample = [results[batch][:lens[batch]] for batch in range(max_batch_size)]
                    z_sample_in_batch, zlens_sample_in_batch = validate_zlen_and_pad(z_sample, lens)
                    # samples_list_batch = [(z_sample_in_batch[batch]) for batch in range(max_batch_size)]
                    # samples_lens_batch = [(zlens_sample_in_batch[batch]) for batch in range(max_batch_size)]
                    samples_list.append(z_sample_in_batch[0])
                    samples_lens_list.append(zlens_sample_in_batch[0])
                    
                if args.is_uniq:
                    samples_tensor = pad_sequence(samples_list, batch_first=True, padding_value=0).transpose(1, -1)
                    samples_lens_tensor = torch.tensor(samples_lens_list, dtype=torch.int)
                
                    reshaped_samples = samples_tensor.view(-1, max_sample_len)
                    unique_samples, indices = torch.unique(reshaped_samples, dim=0, return_inverse=True)
                    samples_list = unique_samples
                # if unique_samples.shape[0] == 2:
                #     print (unique_samples)

            q_out.put(
                (
                    key,
                    {
                        bid: (hypo[0].size(0), 
                              '' if len(hypo[0].tolist())==0 else tokenizer.decode(hypo[0][hypo[0] != 0].tolist())
                              )
                        for bid, hypo in enumerate(
                            zip(samples_list))
                    },
                ),
                block=True,
            )

            del batch
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
    model.clean_unpickable_objs()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model = coreutils.load_checkpoint(model, checkpoint)
    model.eval()
    return model

def _sample(probs, lx, n_samples=None):
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
    

def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="CTC decoder.", training=False, isddp=False
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
        default="cat.ctc.train",
        help="Tell where to import build_model() function. defautl: cat.ctc.train",
    )
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
