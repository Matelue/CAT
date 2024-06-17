# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com) /
# This script prepares phoneme-based lexicon and corrects it for Russian.

dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
# Generating lexicon
  g2ps=/home/mate/cat_multilingual/egs/cv-lang10/local/g2ps/models
  phonetisaurus-apply --model $g2ps/russian.fst --word_list $dict_dir/ru/word_list > $dict_dir/ru/lexicon.txt

# Lexicon correction
cat $dict_dir/ru/lexicon.txt | awk '{$1=""; print $0}' > $dict_dir/ru/phone.txt
