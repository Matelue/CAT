# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Indonesian.

# Generating lexicon
dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
echo " G2P Conversion, generating lexicon"
  g2ps=local/g2ps/models 
  phonetisaurus-apply --model $g2ps/Indonesian.fst --word_list $dict_dir/id/word_list > $dict_dir/id/lexicon.txt

# Lexicon correction
cat $dict_dir/id/lexicon.txt | awk '{$1=""; print $0}' > $dict_dir/id/phone.txt