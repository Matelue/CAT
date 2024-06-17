#!/bin/bash

# Copyright 2024 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script includes the process of data preparation, model training and decoding.

set -e -u
<<"PARSER"
("langs", type=str, nargs='?', default='en', help="language name.")
("dir", type=str, nargs='?', default=$(dirname $0),
    help="Input file.")
("--sta", type=int, default=0,
    help="Start stage. Default: 0")
("--sto", type=int, default=7,
    help="Stop stage. Default: 7")
PARSER
eval $(python utils/parseopt.py $0 $*)

KALDI_ROOT="/opt/kaldi"
export KALDI_ROOT=$KALDI_ROOT
function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}
function get_train_tran() {
    echo $(
        python -c \
            "import json;print(json.load(open('data/metainfo.json'))['$1']['trans'])"
    )
}

stage=$sta
stop_stage=$sto
lang_dir=dict  # lang dir including lexicon and word list
# mode=BPE  # flat_phone or BPE, For BPE mode, word list file is needed
mode=flat_phone

echo "### exp dir: $dir"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "### stage 0: preparing data ..."
    bash local/data_prep.sh $langs
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "### stage 1: preparing tokenizer ..."
    python utils/pipeline/asr.py $dir --sto 1
    echo "### prepare tokenizer finished!"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "### stage 2: packing data ..."
    python utils/pipeline/asr.py $dir --sta 2 --sto 2
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "### stage 3: training model ..."
    python utils/pipeline/asr.py $dir --sta 3 --sto 3
    echo "### training model finished ..."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "### stage 4: CTC decoding ..."
    python utils/pipeline/asr.py $dir --sta 4
    echo "### Decoding finished ..."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for lg in $langs; do
        echo "### stage 5: preparing decode $lg LM with arpa format ..."
        lm_dir="$lang_dir/$lg/lm"
        lm="$lm_dir/4gram.arpa"
        bash utils/pipeline/ngram.sh $lm_dir \
            -o 4 --arpa --output $lm --stop-stage 3
        echo "### Decode LM saved at $lm"
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    for lg in $langs; do
        echo "### stage 6: building $lg decoding graph TLG.fst ..."
        if [ $mode == "BPE" ];then
          awk '{print $1}' $lang_dir/$lg/lexicon.txt > $lang_dir/$lg/word_list
          word_list=$lang_dir/$lg/word_list
          bash utils/tool/build_decoding_graph.sh --word_list $word_list \
            $(get_tokenizer $dir) \
            $(get_tokenizer $lang_dir/$lg/lm) \
            $lang_dir/$lg/lm/4gram.arpa $lang_dir/$lg/graph_bpe
        else
          bash utils/tool/build_decoding_graph.sh \
            $(get_tokenizer $dir) \
            $(get_tokenizer $lang_dir/$lg/lm) \
            $lang_dir/$lg/lm/4gram.arpa $lang_dir/$lg/graph_phn
        fi
        echo "### $lg TLG.fst finish"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for lg in $langs; do
        echo "### stage 7: $lg FST decoding ..."
        score=(0.8 0.9 1.0)
        if [ $mode == "BPE" ];then
            graph_dir=$lang_dir/$lg/graph_bpe
        else
            graph_dir=$lang_dir/$lg/graph_phn
        fi
        for lm in ${score[@]}; do
            bash local/eval_fst_decode.sh \
                $dir \
                $graph_dir \
                --data test_${lg} \
                        --acwt 1.0 --lmwt $lm \
                --mode wer -f
        done
        echo "### $lg fst decode finshed!"
    done
fi
exit 0
