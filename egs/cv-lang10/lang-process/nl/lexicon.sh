# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Dutch.

# Generating lexicon
dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
echo " G2P Conversion, generating lexicon"
  g2ps=local/g2ps/models 
  phonetisaurus-apply --model $g2ps/dutch.fst --word_list $dict_dir/nl/word_list > $dict_dir/nl/lexicon.txt

# Lexicon correction
cat $dict_dir/nl/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/dʒ/d͡ʒ/g; s/œɪ/œ y/g; s/ɛɪ/ɛ i/g' > $dict_dir/nl/phone.txt
