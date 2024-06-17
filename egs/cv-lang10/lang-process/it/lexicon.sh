# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Italian.

dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
# Generating lexicon
  g2ps=/home/mate/cat_multilingual/egs/cv-lang10/local/g2ps/models
  phonetisaurus-apply --model $g2ps/italian_8_2_3.fst --word_list $dict_dir/it/word_list > $dict_dir/it/lexicon.txt

# Lexicon correction
cat $dict_dir/it/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/dʒ/d͡ʒ/g; s/dz/d͡z/g; s/tʃ/t͡ʃ/g; s/ts/t͡s/g; s/∅/ø/g' > $dict_dir/it/phone.txt
