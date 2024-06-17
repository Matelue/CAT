# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Swedish.

# Generating lexicon
dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
echo " G2P Conversion, generating lexicon"
  g2ps=/home/mate/cat_multilingual/egs/cv-lang10/local/g2ps/models
  phonetisaurus-apply --model $g2ps/swedish_4_4_4.fst --word_list $dict_dir/sv-SE/word_list > $dict_dir/sv-SE/lexicon.txt

# Lexicon correction
cat $dict_dir/sv-SE/lexicon.txt | awk '{$1=""; print $0}'  > $dict_dir/sv-SE/phone.txt
