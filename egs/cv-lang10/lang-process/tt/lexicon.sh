# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Tatar.

# Generating lexicon
dict_dir=$1
  g2ps=g2ps/models 
  phonetisaurus-apply --model $g2ps/tatar_2_2_2.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/jo/j o/g; s/g/ɡ/g' > $dict_dir/phone.txt