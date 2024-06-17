# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for English.

dict_dir=/home/mate/cat_multilingual/egs/cv-lang10/dict
# Generating lexicon
  g2ps=/home/mate/cat_multilingual/egs/cv-lang10/local/g2ps/models
  phonetisaurus-apply --model $g2ps/american-english.fst --word_list $dict_dir/en/word_list > $dict_dir/en/lexicon.txt

# Lexicon correction
cat $dict_dir/en/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/ˌ//g; s/l̩/l/g; s/n̩/n/g; s/#//g; s/[.]//g; s/g/ɡ/g; s/ei/e i/g; s/aɪ/a ɪ/g; s/ɔi/ɔ i/g; s/oʊ/o ʊ/g; s/aʊ/a ʊ/g; s/ɔɪ/ɔ ɪ/g; s/ɑɪ/ɑ ɪ/g; s/ɝ/ɜ/g; s/ɚ/ə/g; s/tʃ/t͡ʃ/g; s/dʒ/d͡ʒ/g; s/d ʒ/d͡ʒ/g' > $dict_dir/en/phone.txt
