# Crosslingual ASR model
## Crosslingual fine-tuning with phoneme or subword based pre-trained model
For supervised model, we fine-tune our pre-trained phoneme-based multilingual ASR model [Multi. phoneme S](../Multilingual/Multi._phoneme_S/readme.md) and subword-based multilingual ASR model [Multi. subword](../Multilingual/Multi._subword/readme.md) on Indonesian and Polish Common voice 11.0 datasets. Indonesian and Polish are unseen languages of the two pretrained models. Three training strategies are applied to fine-tune the models on four different duration datasets, including 10 minutes, 1 hour, 10 hours and full data.

| Language | Language ID | # of phonemes |Train hours | Dev hours | Test hours |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `Indonesian` | `id` | 35 | 20.8 | 3.7 | 4.1 |
| `Polish` | `pl` | 35 | 129.9 | 11.4 | 11.5 |

### (1) Phoneme PT and phoneme FT
The phoneme-based supervised pre-trained model [Multi. phoneme S](../Multilingual/Multi._phoneme_S/readme.md) is fine-tuned with __phoneme__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of phonemes. There are some phonemes that are not covered by the pretrained model. To share more phoneme knowledge, we initialize the parameters of the coverd phonemes with pretrained model and randomly initialize for uncovered phonemes.
* %WER of fine-tuning results
    | FT language | 10 minutes | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ | ------ |
    | Indonesian | [6.06](./id/Multi._phoneme_ft_phoneme_10m/readme.md) | [3.27](./id/Multi._phoneme_ft_phoneme_1h/readme.md) | [2.54](./id/Multi._phoneme_ft_phoneme_10h/readme.md) | [2.43](./id/Multi._phoneme_ft_phoneme_20h/readme.md) |
    | Polish | [9.68](./pl/Multi._phoneme_ft_phoneme_10m/readme.md) | [6.95](./pl/Multi._phoneme_ft_phoneme_1h/readme.md) | [5.27](./pl/Multi._phoneme_ft_phoneme_10h/readme.md) | [4.30](./pl/Multi._phoneme_ft_phoneme_130h/readme.md) |

### (2) Phoneme PT and subword FT
The phoneme-based supervised pre-trained model [Multi. phoneme S](../Multilingual/Multi._phoneme_S/readme.md) is fine-tuned with __subword__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of subwords. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 10 minutes | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ | ------ |
    | Indonesian | [98.65](./id/Multi._phoneme_ft_subword_10m/readme.md) | [24.57](./id/Multi._phoneme_ft_subword_1h/readme.md) | [3.59](./id/Multi._phoneme_ft_subword_10h/readme.md) | [2.92](./id/Multi._phoneme_ft_subword_20h/readme.md) |
    | Polish | [80.65](./pl/Multi._phoneme_ft_subword_10m/readme.md) | [8.63](./pl/Multi._phoneme_ft_subword_1h/readme.md) | [4.83](./pl/Multi._phoneme_ft_subword_10h/readme.md) | [3.82](./pl/Multi._phoneme_ft_subword_130h/readme.md) |

### (3) Subword PT and subword FT
 The subword-based supervised pre-trained model [Multi. subword](../Multilingual/Multi._subword/readme.md) is fine-tuned with __subword__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of subwords. There are some subwords that are not covered by the pretrained model. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 10 minutes | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ | ------ |
    | Indonesian | [87.75](./id/Multi._subword_ft_subword_10m/readme.md) | [23.56](./id/Multi._subword_ft_subword_1h/readme.md) | [3.91](./id/Multi._subword_ft_subword_10h/readme.md) | [3.07](./id/Multi._subword_ft_subword_20h/readme.md) |
    | Polish | [52.52](./pl/Multi._subword_ft_subword_10m/readme.md) | [9.16](./pl/Multi._subword_ft_subword_1h/readme.md) | [4.89](./pl/Multi._subword_ft_subword_10h/readme.md) | [3.76](./pl/Multi._subword_ft_subword_130h/readme.md) |

## Crosslingual fine-tuning with self-supervised model
For self-supervised model, we fine-tune the public pretrained English ASR model [Wav2vec-base](https://huggingface.co/facebook/wav2vec2-base/tree/main) and our pretrained multilingual ASR model [Wav2vec-lang10](../Multilingual/Wav2vec-lang10/readme.md) on Indonesian and Polish Common voice 11.0 datasets. Polish and Indonesian are unseen languages of the two pretrained models. Four training strategies are applied to fine-tune the models on three different duration datasets, including 1 hour, 10 hours and full data.

### (1) Wav2vec-En PT and phoneme FT
The pretrained English ASR model [Wav2vec-base](https://huggingface.co/facebook/wav2vec2-base/tree/main) is fine-tuned with __phoneme__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of phonemes. There are some phonemes that are not covered by the pretrained model. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [6.73](./id/Wav2vec-En_ft_phoneme_1h/readme.md) | [3.31](./id/Wav2vec-En_ft_phoneme_10h/readme.md) | [2.83](./id/Wav2vec-En_ft_phoneme_20h/readme.md) |
    | Polish | [11.09](./pl/Wav2vec-En_ft_phoneme_1h/readme.md) | [6.75](./pl/Wav2vec-En_ft_phoneme_10h/readme.md) | [4.57](./pl/Wav2vec-En_ft_phoneme_130h/readme.md) |

### (2) Wav2vec-En PT and subword FT
The pretrained English ASR model [Wav2vec-base](https://huggingface.co/facebook/wav2vec2-base/tree/main) is fine-tuned with __subword__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of subwords. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [100](./id/Wav2vec-En_ft_subword_1h/readme.md) | [5.28](./id/Wav2vec-En_ft_subword_10h/readme.md) | [3.59](./id/Wav2vec-En_ft_subword_20h/readme.md) |
    | Polish | [100](./pl/Wav2vec-En_ft_subword_1h/readme.md) | [7.08](./pl/Wav2vec-En_ft_subword_10h/readme.md) | [3.85](./pl/Wav2vec-En_ft_subword_130h/readme.md) |

### (3) Wav2vec-lang10 PT and phoneme FT
The pretrained multilingual ASR model [Wav2vec-lang10](../Multilingual/Wav2vec-lang10/readme.md) is fine-tuned with __phoneme__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of phonemes. There are some phonemes that are not covered by the pretrained model. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [3.75](./id/Wav2vec-lang10_ft_phoneme_1h/readme.md) | [2.79](./id/Wav2vec-lang10_ft_phoneme_10h/readme.md) | [2.47](./id/Wav2vec-lang10_ft_phoneme_20h/readme.md) |
    | Polish | [7.94](./pl/Wav2vec-lang10_ft_phoneme_1h/readme.md) | [5.65](./pl/Wav2vec-lang10_ft_phoneme_10h/readme.md) | [4.44](./pl/Wav2vec-lang10_ft_phoneme_130h/readme.md) |

### (4) Wav2vec-lang10 PT and subword FT
The pretrained multilingual ASR model [Wav2vec-lang10](../Multilingual/Wav2vec-lang10/readme.md) is fine-tuned with __subword__ labels. For fine-tuning experiment, the output layer of the initialized model need to be matched to the output dimensions of the corresponding language, which is determined by the number of subwords. We initialize the parameters of the output layer randomly.
* %WER of fine-tuning results
    | FT language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [100](./id/Wav2vec-lang10_ft_subword_1h/readme.md) | [4.52](./id/Wav2vec-lang10_ft_subword_10h/readme.md) | [3.15](./id/Wav2vec-lang10_ft_subword_20h/readme.md) |
    | Polish | [100](./pl/Wav2vec-lang10_ft_subword_1h/readme.md) | [5.71](./pl/Wav2vec-lang10_ft_subword_10h/readme.md) | [3.45](./pl/Wav2vec-lang10_ft_subword_130h/readme.md) |