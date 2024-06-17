# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for Polish dataset from CommonVoice

echo "Text Normalization"
data_dir=$/home/mate/cat_multilingual/egs/cv-lang10/data
    for set in dev test excluded_train; do
      paste $data_dir/pl/$set/text > $data_dir/pl/$set/text.bak
      cut <$data_dir/pl/$set/text.bak -f 2- | \
         sed -e 's/,/ /g; s/"/ /g; s/“/ /g; s/[;]/ /g; s/[—]/ /g; s/[.]/ /g; s/:/ /g; s/!/ /g; s/”/ /g; s/?/ /g; s/«/ /g; s/»/ /g' | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/pl/$set/text.trans.tmp
      cut <$data_dir/pl/$set/text.bak -f 1 > $data_dir/pl/$set/text.id.tmp
      paste $data_dir/pl/$set/text.{id,trans}.tmp > $data_dir/pl/$set/text
      cat $data_dir/pl/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/pl/$set/text_new
      mv $data_dir/pl/$set/text_new $data_dir/pl/$set/text
      rm -rf $data_dir/pl/$set/text.{id,trans}.tmp
    done