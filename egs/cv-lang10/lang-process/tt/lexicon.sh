# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Tatar.

# Generating lexicon
dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
echo " G2P Conversion, generating lexicon"
  g2ps=local/g2ps/models 
  phonetisaurus-apply --model $g2ps/tatar_2_2_2.fst --word_list $dict_dir/tt/word_list > $dict_dir/tt/lexicon.txt

# Lexicon correction
cat $dict_dir/tt/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/jo/j o/g; s/g/ɡ/g' > $dict_dir/tt/phone.txt