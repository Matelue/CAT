# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for Indonesian dataset from CommonVoice

data_dir=/home/mate/cat_multilingual/egs/cv-lang10/data
echo "Text Normalization"
    for set in dev test excluded_train; do
      paste $data_dir/id/$set/text > $data_dir/id/$set/text.bak
      cut <$data_dir/id/$set/text.bak -f 2- | \
        sed -e ''s/`/ /g; s/¨/ /g; s/~/ /g; s/=/ /g' \
         -e 's/|/ /g; s/°/ /g; s/[-]/ /g; s/[―]/ /g; s/,/ /g; s/[;]/ /g; s/:/ /g; s/!/ /g; s/¡/ /g; s/?/ /g; s/[¿]/ /g; s/′/ /g; s/‐/ /g; s/´´/ /g' \
         -e 's/[.]/ /g; s/·/ /g; s/‘/ /g; s/’/ /g; s/"/ /g; s/“/ /g; s/”/ /g; s/«/ /g; s/»/ /g; s/≪/ /g; s/≫/ /g; s/[{]/ /g; s/„/ /g; s/−/ /g; s/‑/ /g' \
         -e 's/[}]/ /g; s/®/ /g; s/→/ /g; s/ʿ/ /g; s/‧/ /g; s/ʻ/ /g; s/ ⃗/ /g; s/‹/ /g; s/›/ /g; s/_/ /g; s/ʽ//g; s/  / /g' \
         -e 's/[–]/ /g; s/…/ /g' \
         -e "s/\// /g; s/#/ /g; s/&/ & /g; s/´/'/g; s/''/ /g; s/^[']*/ /g; s/[']*$/ /g; s/ '/ /g; s/' / /g; s/\[/ /g; s/\]/ /g" \
         -e 's/&/ /g;s/(/ /g;s/)/ /g;s/\\/ /g;s/—/ /g;s/，/ /g;s/！/ /g;' \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/id/$set/text.trans.tmp
      cut <$data_dir/id/$set/text.bak -f 1 > $data_dir/id/$set/text.id.tmp
      paste $data_dir/id/$set/text.{id,trans}.tmp > $data_dir/id/$set/text
      cat $data_dir/id/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/id/$set/text_new
      mv $data_dir/id/$set/text_new $data_dir/id/$set/text
      rm -rf $data_dir/id/$set/text.{id,trans}.tmp
    done