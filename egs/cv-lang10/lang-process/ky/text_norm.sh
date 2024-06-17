# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for Kirghiz dataset from CommonVoice

data_dir=/home/mate/cat_multilingual/egs/cv-lang10/data
echo "Text Normalization"
    for set in dev test excluded_train; do
      paste $data_dir/ky/$set/text > $data_dir/ky/$set/text.bak
      cut <$data_dir/ky/$set/text.bak -f 2- | \
        sed -e 's/[.]/ /g; s/!/ /g; s/?/ /g; s/—/ /g; s/»/ /g; s/«/ /g' \
                -e 's/“/ /g; s/"/ /g; s/,/ /g; s/”/ /g; s/=/ /g; s/،/ /g; s/؛/ /g; s/٬/ /g' \
                -e "s/’/ /g; s/‘/ /g; s/\।/ /g; s/॥/ /g; s/„/ /g" \
                -e 's/:/ /g; s/[;]/ /g; s/[(]/ /g; s/[)]/ /g; s/[؟]/ /g' \
                -e 's/[\]/ /g; s/[_]/ /g; s/–/ /g; s/−/ /g; s/…/ /g; s/ -/ /g; s/- / /g' \
                -e 's/⅛/ /g' \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/ky/$set/text.trans.tmp
      cut <$data_dir/ky/$set/text.bak -f 1 > $data_dir/ky/$set/text.id.tmp
      paste $data_dir/ky/$set/text.{id,trans}.tmp > $data_dir/ky/$set/text
      cat $data_dir/ky/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/ky/$set/text_new
      mv $data_dir/ky/$set/text_new $data_dir/ky/$set/text
      rm -rf $data_dir/ky/$set/text.{id,trans}.tmp
    done