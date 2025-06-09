# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Top interface of P2G training

Usage:
    python utils/pipeline/p2g.py -h
"""

from typing import *
import pickle
import argparse

# fmt:off
import os
import sys
# after import common_utils, parent path of utils/ in in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from utils.pipeline.common_utils import *
from cat.shared.tokenizer import LexiconTokenizer, SentencePieceTokenizer
# fmt:on


def pack_data_p2g(
    f_scps: Union[List[str], str],
    f_labels: Union[List[str], str],
    f_out: str,
    output_tokenizer,
    input_tokenizer,
    from_given: bool = False,
    isP2G: bool = False,
    use_hgf_tknz: bool = False,
    filter: Optional[str] = None
):
    import numpy as np
    from tqdm import tqdm

    if os.path.isfile(f_out):
        sys.stderr.write(
            sfmt.warn(
                f"file exist: {sfmt.udl(f_out)}, "
                "rm it if you want to update the data.\n",
                pack_data_p2g,
            )
        )
        return

    if isinstance(f_scps, str):
        f_scps = [f_scps]
    if isinstance(f_labels, str):
        f_labels = [f_labels]

    checkExist("f", f_scps + f_labels)
    checkExist("d", os.path.dirname(f_out))

    l_min = 1
    l_max = float("inf")
    if filter is not None:
        assert ":" in filter, sfmt.error(f"invalid filter format {filter}", pack_data_p2g)
        l_bound, u_bound = (i for i in filter.split(":"))
        if l_bound != "":
            l_min = int(l_bound)
        if u_bound != "":
            l_max = int(u_bound)
    # Read label files.
    twrapper_label = TextUtterancesOrdered(f_labels)
    twrapper_scp = TextUtterancesOrdered(f_scps)
    assert len(twrapper_scp) == len(twrapper_label), sfmt.error(
        "f_scp and f_label should match on the # of lines, "
        f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
        pack_data_p2g,
    )
    cnt_frames = 0

    uids = []
    linfo = np.empty(len(twrapper_label), dtype=np.int64)
    inputs = []
    outputs = []
    cnt = 0
    if from_given:
        if isP2G:
            phone_list = input_tokenizer._units
        else:
            phone_list = output_tokenizer._units
    for (uid, lb), (uid1, ark) in tqdm(zip(twrapper_label, twrapper_scp), total=len(twrapper_label), leave=False
    ):
        assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
        if lb == "":
            sfmt.warn(f"skip empty utt: {uid}", pack_data_p2g)
            continue
        if use_hgf_tknz:
            if isP2G:
                if from_given:
                    in_put = ark
                else:
                    in_put = input_tokenizer.decode(input_tokenizer.encode(lb))
                out_put = lb
            else:
                in_put = lb
                if from_given:
                    out_put = ark
                else:
                    out_put = input_tokenizer.decode(input_tokenizer.encode(lb))
        else:
            if isP2G:
                if from_given:
                    in_put = np.asarray([phone_list[phone] for phone in ark.split()], dtype=np.int64)
                else:
                    in_put = np.asarray(input_tokenizer.encode(lb), dtype=np.int64)
                out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
            else:
                in_put = np.asarray(input_tokenizer.encode(lb), dtype=np.int64)
                if from_given:
                    out_put = np.asarray([phone_list[phone] for phone in ark.split()], dtype=np.int64)
                else:
                    out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
            
        if len(out_put) == 0:
            continue
        outputs.append(out_put)
        linfo[cnt] = len(''.join(in_put.split())) if isinstance(in_put,str) else in_put.shape[0]
        inputs.append(in_put)
        uids.append(uid)
        cnt_frames += len(''.join(in_put.split())) if isinstance(in_put,str) else in_put.shape[0]
        cnt += 1

    if cnt == 0:
        sys.stderr.write(sfmt.error("no qualified seq found.\n", pack_data_p2g))
        sys.exit(1)

    # in order to store labels in a ndarray,
    # first I pad all labels to the max length with -1 (this won't take many memory since labels are short compared to frames)
    # then store the length in the last place, such as
    # [0 1 2 3] -> [0 1 2 3 -1 -1 4]
    # then we can access the data via array[:array[-1]]
    cnt_tokens = sum(len(''.join(x.split())) for x in outputs) if isinstance(outputs[0], str) else sum(x.shape[0] for x in outputs)
    max_len_label = max(len(''.join(x.split())) for x in outputs) if isinstance(outputs[0], str) else max(x.shape[0] for x in outputs)
    if not isinstance(outputs[0], str):
        outputs = np.array(
            [
                np.concatenate(
                    (_x, np.array([-1] * (max_len_label - _x.shape[0]) + [_x.shape[0]]))
                )
                for _x in outputs
            ]
        )
    if not isinstance(inputs[0], str):
        max_len_input = max(x.shape[0] for x in inputs)
        inputs = np.array(
            [
                np.concatenate(
                    (_x, np.array([-1] * (max_len_input - _x.shape[0]) + [_x.shape[0]]))
                )
                for _x in inputs
            ]
        )

    with open(f_out, "wb") as fo:
        pickle.dump(
            {
                "label": np.array(outputs, dtype=object),
                "linfo": linfo[:cnt],
                "arkname": np.array(inputs, dtype=object),
                "key": np.array(uids, dtype=object),
            },
            fo,
        )

    cntrm = len(twrapper_label) - cnt
    if cntrm > 0:
        print(f"pack_data_p2g(): remove {cntrm} unqualified sequences.")
    print(f"# of frames: {cnt_frames} | tokens: {cnt_tokens} | seqs: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expdir", type=str, help="Experiment directory.")
    parser.add_argument(
        "--start_stage",
        dest="stage_beg",
        type=int,
        default=1,
        help="Start stage of processing. Default: 1",
    )
    parser.add_argument(
        "--stop_stage",
        dest="stage_end",
        type=int,
        default=-1,
        help="Stop stage of processing. Default: last stage.",
    )
    parser.add_argument(
        "--ngpu", type=int, default=-1, help="Number of GPUs to be used."
    )
    parser.add_argument(
        "--silent", action="store_true", help="Disable detailed messages output."
    )

    args = parser.parse_args()
    s_beg = args.stage_beg
    s_end = args.stage_end
    if s_end == -1:
        s_end = float("inf")

    assert s_end >= 1, f"Invalid stop stage: {s_end}"
    assert s_beg >= 1 and s_beg <= s_end, f"Invalid start stage: {s_beg}"

    cwd = os.getcwd()
    working_dir = args.expdir
    checkExist("d", working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist("f", f_hyper)
    hyper_cfg = readjson(f_hyper)
    if "env" in hyper_cfg:
        for k, v in hyper_cfg["env"].items():
            os.environ[k] = v
    if "commit" not in hyper_cfg:
        log_commit(f_hyper)

    # setting visible gpus before loading cat/torch
    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)

    from cat.shared import tokenizer as tknz

    initial_datainfo()
    datainfo = readjson(F_DATAINFO)

    ############ Stage 1  Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print(sfmt.header("Stage 1 Tokenizer training"))
            fmt = sfmt(sfmt("Tokenizer training: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ""

        hyper_cfg = readjson(f_hyper)
        if "tokenizer" not in hyper_cfg:
            sys.stderr.write(
                sfmt.missing("tokenizer", raiseerror=False)
                + ", skip tokenizer training.\n"
            )
        else:
            train_tokenizer(f_hyper)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print(sfmt.header("Stage 2 Pickle data"))
            fmt = sfmt(sfmt("Pickle data: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ""

        hyper_cfg = readjson(f_hyper)
        assert "data" in hyper_cfg, sfmt.missing("data", sfmt.udl(f_hyper))
        # load tokenizer from file
        assert "tokenizer" in hyper_cfg, sfmt.missing("tokenizer", sfmt.udl(f_hyper))
        assert "file" in hyper_cfg["tokenizer"], sfmt.missing(
            "file", (sfmt.udl(f_hyper), "tokenizer")
        )

        f_tokenizer = hyper_cfg["tokenizer"]["file"]
        checkExist("f", f_tokenizer)
        tokenizer = tknz.load(f_tokenizer)
        f_in_tokenizer = hyper_cfg["input_tokenizer"]["file"]
        checkExist("f", f_in_tokenizer)
        input_tokenizer = tknz.load(f_in_tokenizer)
        is_noisy_data = hyper_cfg["data"]["is_noisy_data"] if "is_noisy_data" in hyper_cfg["data"] else False
        use_hgf_tknz = hyper_cfg["data"]["use_hgf_tknz"] if "use_hgf_tknz" in hyper_cfg["data"] else False
        is_P2G = hyper_cfg["data"]["is_P2G"] if "is_P2G" in hyper_cfg["data"] else False

        data_settings = hyper_cfg["data"]
        if "filter" not in data_settings:
            data_settings["filter"] = None

        d_pkl = os.path.join(working_dir, "pkl")
        os.makedirs(d_pkl, exist_ok=True)
        for dataset in ["train", "dev"]:
            if dataset not in data_settings:
                sys.stderr.write(
                    sfmt.missing(dataset, "data", raiseerror=False) + ", skip.\n"
                )
                continue

            if dataset == "train" or dataset == "dev":
                filter = data_settings["filter"]
            else:
                filter = None

            if isinstance(data_settings[dataset], str):
                data_settings[dataset] = [data_settings[dataset]]
            f_data = []
            for _set in data_settings[dataset]:
                if _set not in datainfo:
                    raise RuntimeError(
                        f"'{_set}' not found. you can configure it manually in {F_DATAINFO}"
                    )
                f_data.append(datainfo[_set])

            pack_data_p2g(
                [_data["scp"] for _data in f_data],
                [_data["trans"] for _data in f_data],
                f_out=os.path.join(d_pkl, dataset + ".pkl"),
                output_tokenizer=tokenizer,
                input_tokenizer=input_tokenizer,
                from_given=is_noisy_data,
                isP2G=is_P2G,
                use_hgf_tknz=use_hgf_tknz
            )
            del f_data

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print(sfmt.header("Stage 3 NN training"))
            fmt = sfmt(sfmt("NN training: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ""

        train_nn(working_dir, fmt)

    ############ Stage 4  Decode ############
    if s_beg <= 4 and s_end >= 4:
        # FIXME: runing script directly from NN training to decoding always producing SIGSEGV error
        if s_beg <= 3:
            os.system(
                " ".join(
                    [
                        sys.executable,  # python interpreter
                        sys.argv[0],  # file script
                        working_dir,
                        "--silent" if args.silent else "",
                        "--start_stage=4",
                        f"--stop_stage={args.stage_end}",
                        f"--ngpu={args.ngpu}",
                    ]
                )
            )
            sys.exit(0)

        if not args.silent:
            print(sfmt.header("Stage 4 Decode"))
            fmt = sfmt(sfmt("Decode: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ""

        hyper_cfg = readjson(f_hyper)
        assert "inference" in hyper_cfg, sfmt.missing("inference", sfmt.udl(f_hyper))

        cfg_infr = hyper_cfg["inference"]

        checkdir = os.path.join(working_dir, D_CHECKPOINT)
        # do model averaging
        if "avgmodel" in cfg_infr and os.path.isdir(checkdir):
            checkpoint = model_average(
                setting=cfg_infr["avgmodel"], checkdir=checkdir, returnifexist=True
            )[0]
        else:
            checkpoint = None

        # infer
        if "infer" in cfg_infr:
            # try to get inference:infer:option
            assert "bin" in cfg_infr["infer"], sfmt.missing(
                "bin", (sfmt.udl(f_hyper), "inference", "infer")
            )
            assert "option" in cfg_infr["infer"], sfmt.missing(
                "option", (sfmt.udl(f_hyper), "inference", "infer")
            )

            infr_option = cfg_infr["infer"]["option"]
            # find checkpoint
            if infr_option.get("resume", None) is None:
                # no avgmodel found, get the best checkpoint
                if checkpoint is None and os.path.isdir(checkdir):
                    checkpoint = model_average(
                        setting={"mode": "best", "num": 1},
                        checkdir=checkdir,
                        returnifexist=True,
                    )[0]
                # the last check, no fallback method, raise warning
                if checkpoint is None:
                    sys.stderr.write(
                        sfmt.missing("resume", ("inference", "infer", "option"), False)
                        + "\n    ... would causing non-initialized evaluation.\n"
                    )
                else:
                    # there's no way the output of model_average() is an invalid path
                    # ... so here we could skip the checkExist()
                    infr_option["resume"] = checkpoint
                    sys.stdout.write(
                        fmt.format(
                            sfmt.set("inference:infer:option:resume", checkpoint)
                        )
                    )
            else:
                sys.stdout.write(
                    fmt.format(
                        "setting 'resume' in inference:infer:option "
                        "would ignore the inference:avgmodel settings."
                    )
                )
                checkpoint = infr_option["resume"]
                checkExist("f", checkpoint)

            if "config" not in infr_option:
                infr_option["config"] = os.path.join(working_dir, F_NN_CONFIG)
                checkExist("f", infr_option["config"])

            intfname = cfg_infr["infer"]["bin"]
            # check tokenizer
            if intfname != "cat.ctc.cal_logit":
                if "tokenizer" not in infr_option:
                    assert (
                        hyper_cfg.get("tokenizer", {}).get("file", None) is not None
                    ), (
                        "\nyou should set at least one of:\n"
                        f"1. set tokenizer:file ;\n"
                        f"2. set inference:infer:option:tokenizer \n"
                    )
                    infr_option["tokenizer"] = hyper_cfg["tokenizer"]["file"]

            ignore_field_data = False
            os.makedirs(f"{working_dir}/{D_INFER}", exist_ok=True)
            if intfname == "cat.ctc.cal_logit":
                if "input_scp" in infr_option:
                    ignore_field_data = True

                if "output_dir" not in infr_option:
                    assert not ignore_field_data
                    infr_option["output_dir"] = os.path.join(
                        working_dir, D_INFER + "/{}/ark"
                    )
                    sys.stdout.write(
                        fmt.format(
                            sfmt.set(
                                "inference:infer:option:output_dir",
                                infr_option["output_dir"],
                            )
                        )
                    )
            elif intfname in ["cat.ctc.decode", "cat.rnnt.decode", "cat.ctc.decode_g2p", "cat.ctc.decode_phn", "cat.aed.decode", "cat.aed.decode_mt5", "cat.aed.decode_mt5_tkm"]:
                if "input_scp" in infr_option:
                    ignore_field_data = True
                if "output_prefix" not in infr_option:
                    topo = infr_option.get("topo", intfname.split(".")[1])
                    assert (
                        not ignore_field_data
                    ), f"error: seem you forget to set 'output_prefix'"

                    # rm dirname and '.pt'
                    if checkpoint is None:
                        suffix_model = "none"
                    else:
                        suffix_model = os.path.basename(checkpoint).removesuffix(".pt")
                    prefix = (
                        f"{topo}_bs{infr_option.get('beam_size', 'dft')}_{suffix_model}"
                    )
                    if "unified" in infr_option and infr_option["unified"]:
                        prefix += f"_streaming_{infr_option.get('streaming', 'false')}"
                    # set output format
                    a = infr_option.get("alpha", 0)
                    b = infr_option.get("beta", 0)
                    if not (a == 0 and b == 0):
                        prefix += f"_elm-a{a}b{b}"
                    if topo == "rnnt":
                        ilmw = infr_option.get("ilm_weight", 0)
                        if ilmw != 0:
                            prefix += f"_ilm{ilmw}"
                    
                    infr_option["output_prefix"] = os.path.join(
                        working_dir, f"{D_INFER}/" + "{}" + f"/{prefix}"
                    )
            else:
                ignore_field_data = True
                sys.stderr.write(
                    sfmt.warn(
                        f"interface '{intfname}' only support handcrafted execution.\n"
                    )
                )
            import importlib

            interface = importlib.import_module(intfname)
            assert hasattr(
                interface, "main"
            ), f"{intfname} module does not have method main()"
            assert hasattr(
                interface, "_parser"
            ), f"{intfname} module does not have method _parser()"
            if ignore_field_data:
                interface.main(parse_args_from_var(interface._parser(), infr_option))
            else:
                assert "data" in hyper_cfg, sfmt.missing("data", sfmt.udl(f_hyper))
                assert "test" in hyper_cfg["data"], sfmt.missing(
                    "test", (sfmt.udl(f_hyper), "data")
                )

                testsets = hyper_cfg["data"]["test"]
                is_P2G = hyper_cfg["data"]["is_P2G"] if "is_P2G" in hyper_cfg["data"] else False
                if isinstance(testsets, str):
                    testsets = [testsets]
                f_scps = [datainfo[_set]["scp"] for _set in testsets]
                checkExist("f", f_scps)

                running_option = infr_option.copy()
                running_option["is_P2G"] = is_P2G
                for _set, scp in zip(testsets, f_scps):
                    for k in infr_option:
                        if isinstance(infr_option[k], str) and "{}" in infr_option[k]:
                            running_option[k] = infr_option[k].format(_set)
                            sys.stdout.write(
                                fmt.format(f"{_set}: " + sfmt.set(k, running_option[k]))
                            )
                    running_option["input_scp"] = scp
                    if intfname in ["cat.ctc.decode", "cat.rnnt.decode", "cat.ctc.decode_g2p", "cat.ctc.decode_phn", "cat.aed.decode", "cat.aed.decode_mt5", "cat.aed.decode_mt5_tkm"]:
                        if os.path.isfile(running_option["output_prefix"]):
                            sys.stderr.write(
                                sfmt.warn(
                                    f"{sfmt.udl(running_option['output_prefix'])} exists, skip.\n"
                                )
                            )
                            continue

                    # FIXME: this canonot be spawned via mp_spawn, otherwise error would be raised
                    #        possibly due to the usage of mp.Queue
                    interface.main(
                        parse_args_from_var(interface._parser(), running_option)
                    )
        else:
            infr_option = {}

        # compute wer/cer
        if "er" in cfg_infr:
            import utils.wer as wercal

            err_option = cfg_infr["er"]
            if "per" in err_option and err_option["per"]:
                if "dec_tokenizer" in infr_option:
                    err_option["tokenizer"] = infr_option["dec_tokenizer"]
                else:
                    err_option["tokenizer"] = hyper_cfg["tokenizer"]["file"]
            
            if "hy" not in err_option:
                assert (
                    infr_option.get("output_prefix", None) is not None
                ), "inference:er:hy is not set and cannot be resolved from inference:infer."

                err_option["hy"] = infr_option["output_prefix"]
                if err_option.get("oracle", False):
                    err_option["hy"] = err_option["hy"] + ".nbest"

            if "{}" in err_option["hy"]:
                # input in format string
                testsets = hyper_cfg.get("data", {}).get("test", [])
                if isinstance(testsets, str):
                    testsets = [testsets]
                for _set in testsets:
                    sys.stdout.write(f"{_set}\t")
                    wercal.main(
                        parse_args_from_var(
                            wercal._parser(),
                            err_option,
                            [datainfo[_set]["trans"], err_option["hy"].format(_set)],
                        )
                    )
                    sys.stdout.flush()
            else:
                assert "gt" in err_option, sfmt.missing(
                    "gt", (sfmt.udl(f_hyper), "inference", "er")
                )
                wercal.main(
                    parse_args_from_var(
                        wercal._parser(),
                        err_option,
                        [err_option["gt"], err_option["hy"]],
                    )
                )
