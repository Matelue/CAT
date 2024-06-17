# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Kirghiz.

# Generating lexicon
dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
echo " G2P Conversion, generating lexicon"
  g2ps=local/g2ps/models 
  phonetisaurus-apply --model $g2ps/kirghiz_8_2_2.fst --word_list $dict_dir/ky/word_list > $dict_dir/ky/lexicon.txt

# Lexicon correction
cat $dict_dir/ky/lexicon.txt | awk '{$1=""; print $0}' > $dict_dir/ky/phone.txt