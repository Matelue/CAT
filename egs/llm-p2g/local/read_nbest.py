f_path_in = "/mnt/workspace/mate/CAT/egs/commonvoice/exp/llm/02_mt5_pooling/test/test_pl_293.nbest"
f_path_out = "/mnt/workspace/mate/CAT/egs/commonvoice/exp/p2g_exp/79_p2g_mt5_FT_pl_noisy_mul-fted_rag_bm8/decode/test_pl/aed_bs1_last-3_.nbest_"
f_to_read = ""

trans_text = "/mnt/nas4_workspace/spmiData/asr/data/pl/test/text"

save_in_n_files = True
mode = "beamsearch"
# mode = "sample"

import pickle

if mode =="beamsearch":
    if save_in_n_files:
        with open(f_path_in, "rb") as fi:
            data = pickle.load(fi)
            _dataset = list(pickle.load(fi).items())
            for index in range(len(_dataset)):
                okey = _dataset[index][0]
                for nid, (_score, _trans) in _dataset[index][1].items():
                    f_out = f_path_out + "_" + str(nid)
                    with open(f_out, "a", encoding='utf-8') as fo:
                        fo.write(f"{okey}\t{_trans}\n")
    else:
        with open(f_path_in, "rb") as fi:
            _dataset = list(pickle.load(fi).items())
        with open(f_path_out, "w", encoding='utf-8') as fo:
            for index in range(len(_dataset)):
                okey = _dataset[index][0]
                for nid, (_score, _trans) in _dataset[index][1].items():
                    fo.write(f"{okey}\t{_trans}\n")
        text = {}
        f_path_text = f_path_out + "_text"
        with open(trans_text, "r") as ti:
            for line in ti:
                key, value = line.strip().split(maxsplit=1)
                text[key] = value
        with open(f_path_text, 'a') as fc:
            with open(f_path_out, "r") as fi:
                uids_to_keep = [line.split()[0] for line in fi]
                for uid in uids_to_keep:
                    fc.write(f"{uid}\t{text[uid]}\n")
else:
    f_out_nsamp = f_path_out + "_samp"
    f_path_text = f_path_out + "_text"
    with open(f_out_nsamp, "a", encoding='utf-8') as fo:
        with open(f_path_in, "rb") as fi:
            _dataset = list(pickle.load(fi).items())
            keys, scores, trans = [], [], []
            for index in range(len(_dataset)):
                okey = _dataset[index][0]
                for nid, (_score, _trans) in _dataset[index][1].items():
                    fo.write(f"{okey}\t{_trans}\n")
    text = {}
    with open(trans_text, "r") as ti:
        for line in ti:
            key, value = line.strip().split(maxsplit=1)
            text[key] = value
    if f_to_read == "":
        f_to_read = f_out_nsamp
    with open(f_path_text, 'a') as fc:
        with open(f_to_read, "r") as fi:
            uids_to_keep = [line.split()[0] for line in fi]
            for uid in uids_to_keep:
                fc.write(f"{uid}\t{text[uid]}\n")