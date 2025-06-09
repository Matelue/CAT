# Crosslingual Fine-tuning Whistle model

German and Polish are unseen languages for crosslingual ASR. The training data from an unseen language is divided into two scales to simulate different resource scenarios, while the test and validation data remain unchanged. The experiments are also the baseline of our LLM-P2G.



## Crosslingual fine-tuning Whistle model with phoneme or subword labels
Over the CV-Lang10 dataset, we obtain the phoneme-based supervised pre-trained model [Whistle phoneme S](../../../cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md), which can be further fine-tuned with either phoneme labels or subword labels. All the experimental results presented below indicate the word error rate (WER).

## Results
### WER of Whistle phoneme FT (Baseline)

| FT language | 20 hours | 130 hours |
| ------ | ------ | ------ |
| | w LM | w LM |
| Polish  | [16.27](../Crosslingual/pl/Whistle_ft_phoneme_20h/readme.md)  | [4.30](../Crosslingual/pl/Whistle_ft_phoneme_130h/readme.md) |
| German  | [30.71](../Crosslingual/de/Whistle_ft_phoneme_20h/readme.md) | [15.73](../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md) |

### WER of Whistle subword FT (Baseline)

| FT language | 20 hours | | 130 hours | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| Polish | [17.59](../Crosslingual/pl/Whistle_ft_subword_20h/readme.md) | [13.84](../Crosslingual/pl/Whistle_ft_subword_20h/readme.md) | [5.84](../Crosslingual/pl/Whistle_ft_subword_130h/readme.md) | [3.82](../Crosslingual/pl/Whistle_ft_subword_130h/readme.md) |
| German | [27.78](../Crosslingual/de/Whistle_ft_subword_20h/readme.md) | [28.04](../Crosslingual/de/Whistle_ft_subword_20h/readme.md) | [14.09](../Crosslingual/de/Whistle_ft_subword_130h/readme.md) | [14.01](../Crosslingual/de/Whistle_ft_subword_130h/readme.md) |


