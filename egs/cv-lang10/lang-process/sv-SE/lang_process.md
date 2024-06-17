# Swedish
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in [`Swedish_alien_sentences.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/Swedish_alien_sentences.txt).

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __Swedish__ is in the script named [`text_norm.sh`](./text_norm.sh).


## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Swedish__](https://github.com/uiuc-sst/g2ps/blob/masterSwedish/Swedish_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh). 

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed phonemes | Note |
| ------ | ------ |
| `ː` | Accent symbols |
| `ˈ` | Long vowel |
| `ʲ` | Velarization |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_sv-SE.txt`](/home/mate/cat_multilingual/egs/cv-lang10/dict/sv-SE/lexicon_sv-SE.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](/home/mate/cat_multilingual/egs/cv-lang10/dict/sv-SE/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterSwedish/Swedish_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/swed1254. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Swedish__, we choose the first one as the main reference for phoneme checking, which is [PH 1150](https://phoible.org/inventories/view/1150).

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

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 120pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɪ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɪ.ogg) | [aggress`i`va](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/aggressiva.mp3) | a ɡ j r ɛ s s `ɪ` v ɑ | Incorrect G2P labeling. The phoneme `/j/` is redundant, the correct phoneme labeling should be `/a ɡ r ɛ s s ɪ v ɑ/` |
| [`ʉ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʉ.ogg) | [frukostb`u`ffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f r ɵ k u s t b `ʉ` f f e n |  |
| [`ɑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɑ.ogg) | [avsk`y`r](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/avskyr.mp3) | `ɑ` v ɧ `ʏ` r |  |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.ogg) | [a`gg`ressiva](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/aggressiva.mp3) | a `ɡ` j r ɛ s s ɪ v ɑ | Incorrect G2P labeling. The phoneme `/j/` is redundant, the correct phoneme labeling should be `/a ɡ r ɛ s s ɪ v ɑ/` |
| [`ʏ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʏ.ogg) | [avsk`y`r](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/avskyr.mp3) | ɑ v ɧ `ʏ` r |  |
| [`ɧ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɧ.ogg) | [avs`k`yr](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/avskyr.mp3) | ɑ v `ɧ` ʏ r |  |
| [`ɧ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɧ.ogg) | [abrahamitis`k`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/abrahamitiska.mp3) | ɑ b r ɑ h a m i t i `ɧ` a | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/ɧ/` needs to be corrected to `/k/` |
| [`ɕ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɕ.ogg) | [överför`tj`ust](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/överförtjust.mp3) | œ v e r f ø r `ɕ` ʉ s t |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.ogg) | [`a`ggressiva](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/aggressiva.mp3) | `a` ɡ j r ɛ s s ɪ v ɑ | Incorrect G2P labeling. The phoneme `/j/` is redundant, the correct phoneme labeling should be `/a ɡ r ɛ s s ɪ v ɑ/` |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.ogg) | [a`b`rahamitis`k`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/abrahamitiska.mp3) | ɑ `b` r ɑ h a m i t i ɧ a | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/ɧ/` needs to be corrected to `/k/` |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.ogg) | [applå`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/applåd.mp3) | ɑ p p l o `d` |  |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.ogg) | [öv`e`rförtjust](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/överförtjust.mp3) | œ v `e` r f ø r `ɕ` ʉ s t |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.ogg) | [aggr`e`ssiva](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/aggressiva.mp3) | a ɡ j r `ɛ` s s ɪ v ɑ | Incorrect G2P labeling. The phoneme `/j/` is redundant, the correct phoneme labeling should be `/a ɡ r ɛ s s ɪ v ɑ/` |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.ogg) | [frukostbuffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | `f` r ɵ k u s t b ʉ f f e n |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.ogg) | [abra`h`amitiska](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/abrahamitiska.mp3) | ɑ b r ɑ `h` a m i t i ɧ a | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/ɧ/` needs to be corrected to `/k/` |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.ogg) | [abraham`i`tiska](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/abrahamitiska.mp3) | ɑ b r ɑ h a m `i` t i ɧ a | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/ɧ/` needs to be corrected to `/k/` |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.ogg) | [ajöss](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/ajöss.mp3) | ɑ `j` ø s s | The phoneme `/j/` is not contained in any phoneme tables of Phoible, but there is another phoneme `/ʝ/` with the similiar pronunciation in phoneme tables of Phoible |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.ogg) | [fru`k`ostbuffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f r ɵ `k` u s t b ʉ f f e n |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.ogg) | [app`l`åd](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/applåd.mp3) | ɑ p p `l` o d |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.ogg) | [abrahamitiska](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/abrahamitiska.mp3) | ɑ b r ɑ h a `m` i t i ɧ a | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/ɧ/` needs to be corrected to `/k/` |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.ogg) | [bö`n`or](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/bönor.mp3) | b ø `n` ɔ r |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.ogg) | [byggas](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/byggas.mp3) | b y `ŋ` ŋ a s | Incorrect G2P labeling. The letter `g` is pronounced as `/ɡ/`, so the phoneme `/ŋ/` needs to be corrected to `/ɡ/` |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.ogg) | [appl`å`d](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/applåd.mp3) | ɑ p p l `o` d |  |
| [`ɵ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɵ.ogg) | [frukostbuffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f r `ɵ` k u s t b ʉ f f e n |  |
| [`ø`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ø.ogg) | [överf`ö`rtjust](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/överförtjust.mp3) | œ v e r f `ø` r ɕ ʉ s t |  |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.ogg) | [bön`o`r](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/bönor.mp3) | b ø n `ɔ` r |  |
| [`œ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/œ.ogg) | [`ö`verförtjust](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/överförtjust.mp3) | `œ` v e r f ø r ɕ ʉ s t |  |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.ogg) | [a`pp`l`å`d](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/applåd.mp3) | ɑ `p p` l `o` d |  |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.ogg) | [f`r`ukostbuffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f `r` ɵ k u s t b ʉ f f e n | The phoneme `/r/` is not contained in any phoneme tables of Phoible, but the G2P labeling sounds correct |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.ogg) | [cyan](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/cyan.mp3) | `s` ʏ a n | Incorrect G2P labeling. The letter `c` is pronounced as `/s/` with letter `e`, `i`, `y`,`ä` and `ö` |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.ogg) | [frukos`t`buffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f r ɵ k u s `t` b ʉ f f e n |  |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.ogg) | [fruk`o`stbuffén](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/frukostbuffén.mp3) | f r ɵ k `u` s t b ʉ f f e n |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.ogg) | [a`v`skyr](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/avskyr.mp3) | ɑ `v` ɧ ʏ r |  |
| [`y`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/y.ogg) | [byggas](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/sv-SE/word_audio/byggas.mp3) | b `y` ŋ ŋ a s | Incorrect G2P labeling. The letter `g` is pronounced as `/ɡ/`, so the phoneme `/ŋ/` needs to be corrected to `/ɡ/` |

