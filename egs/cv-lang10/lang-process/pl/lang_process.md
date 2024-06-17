# Polish
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`Polish_alien_sentences.txt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/data-process/pl/Polish_alien_sentences.txt)

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __Polish__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Polish__](https://github.com/uiuc-sst/g2ps/blob/master/Polish/Polish_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols and long vowel symbols to enable sharing more phonemes between different languages. 
    
| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ʲ` | Velarization |


## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_pl.txt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/lexicon_pl.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/master/Polish/Polish_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/poli1260. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Polish__, we choose the first one as the main reference for phoneme checking, which is [PH 1046](https://phoible.org/inventories/view/1046).

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

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 215pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɕ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɕ.ogg) | [rozw`ś`cieczonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v `ɕ` t͡ɕ ɛ ʈ͡ʂ ɔ n ɨ x | Incorrect G2P labeling. The redundant phonemes `/ɨ/` should be removed |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.ogg) | [chorą`g`wi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/chorągwi.mp3) | x ɔ r ɔ ŋ `ɡ` v | Incorrect G2P labeling. It should be corrected to `/x ɔ r ɔ ɡ v i/` |
| [`ɨ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɨ.ogg) | [rozwśc`i`eczonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ t͡ɕ ɛ ʈ͡ʂ ɔ n `ɨ` x |  |
| [`ɲ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɲ.ogg) | [międzyi`n`stytucjo`n`alnego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | m v i ɛ n d͡ʑ ɨ i `ɲ` s t ɨ t u t͡s j ɔ `ɲ` a l n ɛ ɡ ɔ | Incorrect G2P labeling. The letter `n` is pronounced as `/n/`, so the phoneme `/ɲ/` needs to be corrected to `/n/` |
| [`ʂ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʂ.ogg) | [be`z`przykładne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | b ɛ `ʂ` ɛ p ʐ ɨ k w a d n ɛ | Incorrect G2P labeling. The letter `z` is pronounced as `/z/`, so the phoneme `/ʂ/` needs to be corrected to `/z/` |
| [`ʐ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʐ.ogg) | [bezpr`z`ykładne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | b ɛ ʂ ɛ p `ʐ` ɨ k w a d n ɛ | Incorrect G2P labeling. The letter `z` is pronounced as `/z/`, so the phoneme `/ʐ/` needs to be corrected to `/z/` |
| [`ʑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʑ.ogg) | [fonta`ź`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/fontaź.mp3) | f ɔ n a `ʑ` | Incorrect G2P labeling. It should be corrected to `/f ɔ n t a ʑ/` |
| [`ʈʂ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʈʂ.ogg) | [rozwście`cz`onych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ t͡ɕ ɛ `ʈ͡ʂ` ɔ n ɨ x | The same pronunciation for `/ʈʂ/` and `/ʈ͡ʂ/`, so replacing `/ʈʂ/` with `/ʈ͡ʂ/` |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.ogg) | [font`a`ź](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/fontaź.mp3) | f ɔ n `a` ʑ | Incorrect G2P labeling. It should be corrected to `/f ɔ n t a ʑ/` |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.ogg) | [`b`ezprzykładne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | `b` ɛ ʂ ɛ p ʐ ɨ k w a d n ɛ | Incorrect G2P labeling. It should be corrected to `/b ɛ z p z ɨ k w a d n ɛ/` |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.ogg) | [bezprzykła`d`ne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | b ɛ ʂ ɛ p ʐ ɨ k w a `d` n ɛ | Incorrect G2P labeling. It should be corrected to `/b ɛ z p z ɨ k w a d n ɛ/` |
| [`dʑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʑ.ogg) | [mię`dz`yinstytucjonalnego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | m v i ɛ n `d͡ʑ` ɨ i ɲ s t ɨ t u t͡s j ɔ ɲ a l n ɛ ɡ ɔ | The same pronunciation for `/dʑ/` and `/d͡ʑ/`, so replacing `/dʑ/` with `/d͡ʑ/` |
| [`ɖʐ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɖʐ.ogg) | [azerbej`dż`an](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/azerbejdżan.mp3) | a ʂ ɨ ɛ r b ɛ j `ɖ͡ʐ` a n | The phoneme `/ɖʐ/` is not contained in [PH 1046](https://phoible.org/inventories/view/1046), but contained in [EA 2604](https://phoible.org/inventories/view/2604). And the same pronunciation for `/ɖʐ/` and `/ɖ͡ʐ/`, so replacing `/ɖʐ/` with `/ɖ͡ʐ/` |
| [`dz`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dz.ogg) | [cu`dz`ysłowie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s u `d͡z` ɨ s w ɔ v j ɛ | The same pronunciation for `/dz/` and `/d͡z/`, so replacing `/dz/` with `/d͡z/` |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.ogg) | [rozwści`e`czonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ t͡ɕ `ɛ` ʈ͡ʂ ɔ n ɨ x | The phoneme `/ɛ/` is not contained in [PH 1046](https://phoible.org/inventories/view/1046), but contained in [EA 2604](https://phoible.org/inventories/view/2604) |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.ogg) | [`f`ontaź](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/fontaź.mp3) | `f` ɔ n a ʑ | Incorrect G2P labeling. It should be corrected to `/f ɔ n t a ʑ/` |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.ogg) | [m`i`ędzyinstytucjonalnego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | m v `i` ɛ n d͡ʑ ɨ `i` ɲ s t ɨ t u t͡s j ɔ ɲ a l n ɛ ɡ ɔ | Incorrect G2P labeling. It should be corrected to `/m i ɛ d͡ʑ ɨ i n s t ɨ t u t͡s j ɔ n a l n ɛ ɡ ɔ/` |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.ogg) | [cudzysłow`i`e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s u d͡z ɨ s w ɔ v `j` ɛ | Incorrect G2P labeling. The letter `i` is pronounced as `/i/`, so the phoneme `/j/` needs to be corrected to `/i/` |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.ogg) | [bezprzy`k`ładne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | b ɛ ʂ ɛ p ʐ ɨ `k` w a d n ɛ | Incorrect G2P labeling. It should be corrected to `/b ɛ z p z ɨ k w a d n ɛ/` |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.ogg) | [międzyinstytucjona`l`nego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | m v i ɛ n d͡ʑ ɨ i ɲ s t ɨ t u t͡s j ɔ ɲ a `l` n ɛ ɡ ɔ | Incorrect G2P labeling. It should be corrected to `/m i ɛ d͡ʑ ɨ i n s t ɨ t u t͡s j ɔ n a l n ɛ ɡ ɔ/` |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.ogg) | [`m`iędzyinstytucjonalnego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | `m` v i ɛ n d͡ʑ ɨ i ɲ s t ɨ t u t͡s j ɔ ɲ a l n ɛ ɡ ɔ | Incorrect G2P labeling. It should be corrected to `/m i ɛ d͡ʑ ɨ i n s t ɨ t u t͡s j ɔ n a l n ɛ ɡ ɔ/` |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.ogg) | [rozwścieczo`n`ych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ t͡ɕ ɛ ʈ͡ʂ ɔ `n` ɨ x |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.ogg) | [chorą`g`wi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/chorągwi.mp3) | x ɔ r ɔ `ŋ` ɡ v | Incorrect G2P labeling. The redundant phonemes `/ŋ/` should be removed and the phoneme labeling should be corrected to `/x ɔ r ɔ ɡ v i/` |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.ogg) | [r`o`zwścieczonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r `ɔ` z ɨ v ɕ t͡ɕ ɛ ʈ͡ʂ `ɔ` n ɨ x | The phoneme `/ɔ/` is not contained in [PH 1046](https://phoible.org/inventories/view/1046), but contained in [EA 2604](https://phoible.org/inventories/view/2604) |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.ogg) | [bez`p`rzykładne](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/bezprzykładne.mp3) | b ɛ ʂ ɛ `p` ʐ ɨ k w a d n ɛ | Incorrect G2P labeling. It should be corrected to `/b ɛ z p z ɨ k w a d n ɛ/` |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.ogg) | [cho`r`ągwi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/chorągwi.mp3) | x ɔ `r` ɔ ŋ ɡ v |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.ogg) | [cudzy`s`łowie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s u d͡z ɨ `s` w ɔ v j ɛ |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.ogg) | [międzyins`t`y`t`ucjonalnego](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/międzyinstytucjonalnego.mp3) | m v i ɛ n d͡ʑ ɨ i ɲ s `t` ɨ `t` u t͡s j ɔ ɲ a l n ɛ ɡ ɔ |  |
| [`tɕ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tɕ.ogg) | [rozwś`ci`eczonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ `t͡ɕ` ɛ ʈ͡ʂ ɔ n ɨ x | The same pronunciation for `/tɕ/` and `/t͡ɕ/`, so replacing `/tɕ/` with `/t͡ɕ/` |
| [`ts`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ts.ogg) | [`c`udzysłowie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | `t͡s` u d͡z ɨ s w ɔ v j ɛ | The same pronunciation for `/ts/` and `/t͡s/`, so replacing `/ts/` with `/t͡s/` |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.ogg) | [c`u`dzysłowie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s `u` d͡z ɨ s w ɔ v j ɛ |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.ogg) | [cudzysło`w`ie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s u d͡z ɨ s w ɔ `v` j ɛ |  |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.ogg) | [cudzys`ł`owie](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/cudzysłowie.mp3) | t͡s u d͡z ɨ s `w` ɔ v j ɛ |  |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.ogg) | [rozwścieczony`ch`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ z ɨ v ɕ t͡ɕ ɛ ʈ͡ʂ ɔ n ɨ `x` |  |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.ogg) | [ro`z`wścieczonych](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/word_audio/rozwścieczonych.mp3) | r ɔ `z` ɨ v ɕ t͡ɕ ɛ ʈ͡ʂ ɔ n ɨ x |  |