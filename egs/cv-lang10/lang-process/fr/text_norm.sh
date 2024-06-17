# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for French dataset from CommonVoice

data_dir=$/home/mate/cat_multilingual/egs/cv-lang10/data
echo "Text Normalization"
    for set in dev test excluded_train; do
      paste $data_dir/fr/$set/text > $data_dir/fr/$set/text.bak
      cut <$data_dir/fr/$set/text.bak -f 2- | \
        sed 's/-/ /g; s/ǃ/ /g; s/{/ /g; s/}/ /g; s/|/ /g'\
            -e 's/^/ /g; s/~/ /g; s/®/ /g; s/∆/ /g; s/̲/ /g'\
            -e 's/§/ /g; s/×/ /g; s/|/ /g; s/→/ /g; s/↔/ /g'\
            -e 's/☉/ /g; s/∼/ /g; s/─/ /g; s/―/ /g; s/ʔ/ /g' \
            -e 's/\^/ /g; s/⠈/ /g; s/ĳ/ij/g; s/ǀ/ /g' \
            -e 's/⋅/ /g; s/‐/ /g; s/‹/ /g; s/›/ /g; s/ː/ /g' \
            -e "s/ '/ /g; s/' / /g; s/'$/ /g; s/ ’/ /g; s/’ / /g" \
            -e 's/ ‘/ /g; s/‘ / /g; s/‘$/ /g; s/’$/ /g' \
            -e "s/‘/'/g; s/’/'/g" \
            -e "s/''/'/g" | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/fr/$set/text.trans.tmp
      cut <$data_dir/fr/$set/text.bak -f 1 > $data_dir/fr/$set/text.id.tmp
      paste $data_dir/fr/$set/text.{id,trans}.tmp > $data_dir/fr/$set/text
      cat $data_dir/fr/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/fr/$set/text_new
      mv $data_dir/fr/$set/text_new $data_dir/fr/$set/text
      rm -rf $data_dir/fr/$set/text.{id,trans}.tmp
    done