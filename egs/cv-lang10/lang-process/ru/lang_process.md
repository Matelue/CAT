# Russian
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) Before creating lexicon, we need to normalize text. The code of text normalization for __Russian__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Russian__](https://github.com/uiuc-sst/g2ps/blob/masterRussian/Russian_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ˌ` | Syllable |


## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_ru.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/lexicon_ru.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterRussian/Russian_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/russ1263. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Russian__, we choose the first one as the main reference for phoneme checking, which is [EA 2261](https://phoible.org/inventories/view/2261).

Note that the G2P procedure is not perfect,  the G2P-generated `phone_list.txt` is not exactly the same as the ideal IPA symbol table in LanguageNet. Further, the IPA symbol table in LanguageNet may also differ from other IPA symbol tables from other linguistic resources (e.g., Phoible). So we need to check. The inconsistencies are recorded in the following. The lexicon is not modified, since a complete modification of the whole lexicon requires non-trivial manual labor. The final lexicon is not perfect, with some noise.

### Checking process

For each IPA phoneme in  `phone_list.txt`, its sound obtained from Wikipedia is listened. 
A word, which consists of this IPA phoneme, is arbitrarily chosen from the lexicon and listened from Google Translate.
By comparing these two sounds, we could do phoneme check, which is detailed as follows.

#### Check whether there is any inconsistency between `phone_list.txt`, IPA symbol table in LanguageNet, and IPA symbol table in Phoible
A phoneme in `phone_list.txt` should appear in both the IPA symbol table in LanguageNet G2P and the IPA symbol tables in Phoible.

#### Check whether the G2P labeling is correct
The Wikipedia sound of the phoneme should match that appeared in the corresponding position in the Google Translate pronunciation of the word, which consists of this IPA phoneme.

If either of the above two checks fail, it means that the lexicon contains some errors and needs to be further corrected.

### Checking result

The checking result is shown in the following table. Clicking the hyperlinks will download the sound files for your listening.
* The first column shows the phonemes in `phone_list.txt`.  
* The second and third columns show the word and its G2P labeling. The word's G2P labeling consists of the phoneme in the first column.
* The last column contains some checking remarks.

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 150pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [бла`г`онадежность](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/благонадежность.mp3) | b l a `ɡ` o n a d e ʐ n o s t |  |
| [`ʉ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʉ.mp3) | [брош`ю`ры](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/брошюры.mp3) | b r o ʂ `ʉ` r ɨ | Incorrect G2P labeling. The phoneme `/ʉ/` is not contained in any phoneme tables of Phoible, and it needs to be corrected to `/j u/` after listening |
| [`ɪ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɪ.mp3) | [вас`е`ньке](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/васеньке.mp3) | v a s `ɪ` n k e | Incorrect G2P labeling. The phoneme `/ɪ/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/e/` after listening |
| [`ʂ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʂ.mp3) | [бро`ш`юры](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/брошюры.mp3) | b r o `ʂ` ʉ r ɨ |  |
| [`ɕ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɕ.mp3) | [взаимоукрепляю`щ`их](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e p l æ j u `ɕ` i x |  |
| [`ɨ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɨ.mp3) | [брошюр`ы`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/брошюры.mp3) | b r o ʂ ʉ r `ɨ` |  |
| [`ʐ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʐ.mp3) | [благонаде`ж`ность](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/благонадежность.mp3) | b l a ɡ o n a d e `ʐ` n o s t |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [`а`лёна](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/алёна.mp3) | `a` l ɵ n a |  |
| [`æ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/æ.mp3) | [взаимоукрепл`я`ющих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e p l `æ` j u ɕ i x | Incorrect G2P labeling. The phoneme `/æ/` is not contained in any phoneme tables of Phoible, and it needs to be corrected to `/j a/` after listening |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [`б`рошюры](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/брошюры.mp3) | `b` r o ʂ ʉ r ɨ |  |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [благона`д`ежность](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/благонадежность.mp3) | b l a ɡ o n a `d` e ʐ n o s t |  |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [васеньк`е`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/васеньке.mp3) | v a s ɪ n k `e` |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [мало`э`ффективными](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/малоэффективными.mp3) | m a l o `ɛ` f e k t i v n ɨ m i | The phoneme `/ɛ/` is not contained in [EA 2261](https://phoible.org/inventories/view/2261), but contained in [SPA 166](https://phoible.org/inventories/view/166) |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [малоэ`фф`ективными](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/малоэффективными.mp3) | m a l o ɛ `f` e k t i v n ɨ m i |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.mp3) | [`h`r](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/hr.mp3) | `h` r | Incorrect G2P labeling. The phoneme `/h/` is not contained in any phoneme tables of LanguageNet or Phoible because `hr` is a English word which is pronunced as `/e i t͡ʃ r/` |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [малоэффективным`и`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/малоэффективными.mp3) | m a l o ɛ f e k t i v n ɨ m `i` |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [взаимоукрепля`ю`щих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e p l æ `j` u ɕ i x |  |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [васень`к`е](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/васеньке.mp3) | v a s ɪ n `k` e |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [а`л`ёна](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/алёна.mp3) | a `l` ɵ n a |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [малоэффективны`м`и](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/малоэффективными.mp3) | `m` a l o ɛ f e k t i v n ɨ `m` i |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [алё`н`а](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/алёна.mp3) | a l ɵ `n` a |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [благонадежн`о`сть](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/благонадежность.mp3) | b l a ɡ o n a d e ʐ n `o` s t |  |
| [`ɵ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɵ.mp3) | [ал`ё`на](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/алёна.mp3) | a l `ɵ` n a | The phoneme `/ɵ/` is not contained in any phoneme tables of Phoible, but the G2P labeling is correct after listening |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [взаимоукре`п`ляющих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e `p` l æ j u ɕ i x |  |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.mp3) | [б`р`ошюры](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/брошюры.mp3) | b `r` o ʂ ʉ `r` ɨ |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [ва`с`еньке](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/васеньке.mp3) | v a `s` ɪ n k e |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [благонадежнос`ть`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/благонадежность.mp3) | b l a ɡ o n a d e ʐ n o s `t` |  |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [взаимоукрепля`ю`щих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e p l æ j `u` ɕ i x | The letter `ю` is pronunced as `/j u/` |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [`в`асеньке](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/васеньке.mp3) | `v` a s ɪ n k e |  |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.mp3) | [взаимоукрепляющи`х`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o y k r e p l æ j u ɕ i `x` |  |
| [`y`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/y.mp3) | [взаимо`у`крепляющих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v z a i m o `y` k r e p l æ j u ɕ i x | Incorrect G2P labeling. The phoneme `/y/` is not contained in any phoneme tables of Phoible, and it needs to be corrected to `/u/` |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [в`з`аимоукрепляющих](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ru/word_audio/взаимоукрепляющих.mp3) | v `z` a i m o y k r e p l æ j u ɕ i x |  |



