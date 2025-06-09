# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""
Transducer trainer.
"""

__all__ = ["TransducerTrainer", "build_model", "_parser", "main"]


from ..shared import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import sortedPadCollateMT5, sortedPadCollateMT5G2P, MT5Dataset
import numpy as np
import matplotlib.pyplot as plt

import os
import math
import argparse
import Levenshtein
from typing import *
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
from transformers import modeling_outputs as hgf_oput
import transformers
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from transformers.models.mt5 import MT5ForConditionalGeneration

# NOTE (huahuan):
#   1/4 subsampling is used for Conformer model defaultly
#   for other sampling ratios, you need to modify the value.
#   Commonly, you can use a relatively larger value for allowing some margin.
SUBSAMPLING = 4


def check_label_len_for_ctc(
    tupled_mat_label: Tuple[torch.FloatTensor, torch.LongTensor]
):
    """filter the short seqs for CTC/CRF"""
    return tupled_mat_label[0].shape[0] // SUBSAMPLING > tupled_mat_label[1].shape[0]


def filter_hook(dataset):
    return dataset.select(check_label_len_for_ctc)


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
        # mkwargs["T_dataset"] = KaldiSpeechDataset
        mkwargs["T_dataset"] = MT5Dataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = sortedPadCollateMT5(tknz=args.hgf_tokenizer) if args.is_P2G else sortedPadCollateMT5G2P(tknz=args.hgf_tokenizer) ## 5-11改 mate

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if "_wds_hook" not in mkwargs:
        mkwargs["_wds_hook"] = filter_hook

    if (
        "func_eval" not in mkwargs
        and hasattr(args, "eval_error_rate")
        and args.eval_error_rate
    ):
        mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    # NOTE: for CTC training, the input feat len must be longer than the label len
    #       ... when using webdataset (--ld) to load the data, we deal with
    #       ... the issue by `_wds_hook`; if not, we filter the unqualified utterances
    #       ... before training start.
    # if args.ld is None:
    #     coreutils.distprint(f"> filter seqs by ctc restriction", args.gpu)
    #     tr_dataset = manager.trainloader.dl.dataset
    #     orilen = len(tr_dataset)
    #     tr_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
    #     val_dataset = manager.valloader.dataset
    #     val_orilen = len(val_dataset)
    #     val_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
    #     coreutils.distprint(
    #         f"  filtered {orilen-len(tr_dataset)} utterances.", args.gpu
    #     )

    # training
    manager.run(args)


class MT5Trainer(nn.Module):
    def __init__(self,
                 model: model_zoo.MT5FromPretrainedModel,
                 use_hgf_tknz: bool = True,
                 phn_tknz: str = None,
                 grph_tknz: str = None
                 ):
        super().__init__()

        self.model = model
        self.use_hgf_tknz = use_hgf_tknz
    
    def forward(self, x, lx, y, attn_mask):
        x = x.to(self.model.model.device)
        y = y.to(self.model.model.device)
        attn_mask = attn_mask.to(self.model.model.device)
        if self.use_hgf_tknz:
            
            model_out = self.model(x, y, attn_mask, return_dict=False, return_loss_only=True)
        else:
            model_out = self.model(x, y, attn_mask)

        
        
        ##########·DEBUG·CODE ###########
        if torch.isinf(model_out).any():
            print("feature shape:", x.shape)
            print("Given lengths:", lx.tolist())
            print("labels:", y.tolist())
            raise StopIteration
        ##########·DEBUG·CODE ###########        
        # model_out: loss, logits, past_key_values, encoder_last_hidden_states
        return model_out

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.
    def clean_unpickable_objs(self):
        # CTCBeamDecoder is unpickable,
        # So, this is required for inference.
        self.attach["decoder"] = None

    def register_crf_ctx(self, den_lm: Optional[str] = None):
        """Register the CRF context on model device."""
        assert self.is_crf

        from ctc_crf import CRFContext

        self._crf_ctx = CRFContext(
            den_lm, next(iter(self.encoder.parameters())).device.index
        )

    @torch.no_grad()
    def get_wer(
        self, xs: torch.Tensor, lx: torch.Tensor, ys: torch.Tensor, attn_mask: torch.Tensor
    ):
        # if self.attach["decoder"] is None:
        #     raise RuntimeError(
        #         f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
        #     )

        model_out = self.model(xs, ys, attn_mask, return_dict=False, return_loss_only=False)
        logits = torch.log_softmax(model_out[1], dim = -1, dtype=torch.float32)
        hypos = torch.argmax(logits, dim = -1).tolist()
        gt = ys.tolist()

        return cal_wer(gt, hypos)


    def generate_attention_mask(self, lens_matrix):
        seq_len = torch.max(lens_matrix) 
        attention_mask = torch.arange(seq_len, device=lens_matrix.device)[
            None, :
        ] < lens_matrix[:, None].to(lens_matrix.device)
        # attention_mask = (1.0 - attention_mask.to(self.dtype)) * torch.finfo(self.dtype).min
        return attention_mask.to(self.dtype)

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
        feats, ilens, labels, attn_mask = minibatch[:4]
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        attn_mask = attn_mask.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, ilens, labels, attn_mask)
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
) -> Union[nn.parallel.DistributedDataParallel, MT5Trainer]:
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
        
    if "decoder" in cfg["trainer"]:
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])
    
    assert "model" in cfg["trainer"]
    netconfigs = cfg["trainer"]["model"]
    net_kwargs = netconfigs["kwargs"]
    mt5_model = getattr(model_zoo, netconfigs["type"])(
        **net_kwargs
    ) 

    trainer_cfg = cfg["trainer"]["kwargs"]
    
    model = MT5Trainer(mt5_model, **trainer_cfg)


    # 冻结前面层参数, freeze默认为False,只更新最后三层参数
    if cfg.get("freeze", False):
            model.requires_grad_(False)
            # am_model.cells[13].ffm1.requires_grad_(True)  # conformer cell
            # am_model.cells[13].ln.requires_grad_(True)  # conformer cell
            model.classifier.requires_grad_(True)  # linear layer

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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args["gpu"]])
    return model


def _parser():
    parser = coreutils.basic_trainer_parser("Seq2Seq trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of AED loss (default). "
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
        "    since we import the build_model() function in cat.aed,\n"
        "    we should avoid calling `python -m cat.aed.train`, instead\n"
        "    running `python -m cat.aed`"
    )
