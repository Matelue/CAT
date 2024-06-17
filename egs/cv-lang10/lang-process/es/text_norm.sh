# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for Spanish dataset from CommonVoice

echo "Text Normalization"
data_dir=/home/mate/cat_multilingual/egs/cv-lang10/data
    for set in dev test excluded_train; do
      paste $data_dir/es/$set/text > $data_dir/es/$set/text.bak
      cut <$data_dir/es/$set/text.bak -f 2- | \
        sed -e 's/`/ /g; s/¨/ /g; s/~/ /g; s/=/ /g' \
         -e 's/|/ /g; s/°/ /g; s/[-]/ /g; s/[―]/ /g; s/,/ /g; s/[;]/ /g; s/:/ /g; s/!/ /g; s/¡/ /g; s/?/ /g; s/[¿]/ /g; s/′/ /g; s/‐/ /g; s/´´/ /g' \
         -e 's/[.]/ /g; s/·/ /g; s/‘/ /g; s/’/ /g; s/"/ /g; s/“/ /g; s/”/ /g; s/«/ /g; s/»/ /g; s/≪/ /g; s/≫/ /g; s/[{]/ /g; s/„/ /g; s/−/ /g; s/‑/ /g' \
         -e 's/[}]/ /g; s/®/ /g; s/→/ /g; s/ʿ/ /g; s/‧/ /g; s/ʻ/ /g; s/ ⃗/ /g; s/‹/ /g; s/›/ /g; s/_/ /g; s/ʽ//g; s/￼￼/ /g; s/m̪/m/g; s/ː/ /g; s/ﬁ/fi/g; s/ﬂ/fl/g' | \
         sed -e 's/[–]/ /g; s/…/ /g' | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/es/$set/text.trans.tmp
      cut <$data_dir/es/$set/text.bak -f 1 > $data_dir/es/$set/text.id.tmp
      paste $data_dir/es/$set/text.{id,trans}.tmp > $data_dir/es/$set/text
      cat $data_dir/es/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/es/$set/text_new
      mv $data_dir/es/$set/text_new $data_dir/es/$set/text
      rm -rf $data_dir/es/$set/text.{id,trans}.tmp
    done