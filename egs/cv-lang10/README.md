# cv-lang10
## data
[`data`](./data/) contains the efficient data management file [metainfo.json](./data/metainfo.json).

## lang-process
All of our ASR models are trained with the processed cv-lang10 data covering 12 languages, which are sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. The data processing for each language are detailed in [`lang-process.md`](./lang-process/lang-process.md). For convenience, we adopt ISO-639-1 code to represent language ID, and the 12 languages and training hours are as follow.
| Serial number | Language | Language ID | Training hours |
| ----------- | ----------- | ----------- | ----------- |
| 1 | `English` | `en` | 2227.3 |
| 2 | `Spanish` | `es` | 382.3 |
| 3 | `French` | `fr` | 823.4 |
| 4 | `Italian` | `it` | 271.5 |
| 5 | `Kyrgyz` | `ky` | 32.7 |
| 6 | `Dutch` | `nl` | 70.2 |
| 7 | `Russian` | `ru` | 149.8 |
| 8 | `Swedish` | `sv-SE` | 29.8 |
| 9 | `Turkish` | `tr` | 61.5 |
| 10 | `Tatar` | `tt` | 20.8 |
| 11 | `Polish` | `pl` | 130 |
| 12 | `Indonesian` | `id` | 20.8 |

## local
[`local`](./local/) contains the script [data_prep.md](./local/data_prep.md) preparing data and generating pronunciation lexicon for each language. Besides, there are some useful tools to debug our experiments.

## exp
[`exp`](./exp/) contains configuration files and detailed training process of our models.
### Experiment setup
We adapt the Conformer and CTC to train our models. Three training strategies were applied for comparison, which are monolingual, multilingual and cross-lingual training.

#### [Monolingual](./exp/Monolingual/Monolingual.md)
10 monolingual phoneme-based ASR models are trained on each language dataset seperately and then is evaluated on test dataset of corresponding language whitout fine-tuneing. For Indonesian and Polish, the training data is divided into three scales: 1 hour, 10 hours, and full. And the phoneme-based model and subword-based model are both trained with these scales data seperately.
#### [Multilingual](./exp/Multilingual/Multilingual.md)
3 phoneme-based models of different sizes are trained, including small(90 MB), medium(218 MB) and large(543 MB). And subword-based and wav2vec-based model of small size are also trained for comprison. The multilingual ASR model are trained on cv-lang10 data except Indonesian and Polish and then is evaluated on test dataset of corresponding language whitout fine-tuneing.
#### [Crosslingual](./exp/Crosslingual/Crosslingual.md)
The unseen languages of our multilingual models are Indonesian and Polish. 12 crosslingual models with different fine-tuning strategys are trained for both Indonesian and Polish. All of the Crosslingual models are fine-tuned on the basis of the pretrained multilingual phoneme-based model of small size, subword-based model or wav2vec-based model with the same fine-tuning strategy. The performence of the fine-tuned models are evaluated on cross languages test dataset.

### Results
#### Monolingual and Multilingual
* %PER
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | 90 MB | 7.92 | 2.47 | 4.93 | 2.87 | 2.23 | 5.89 | 2.72 | 16.11 | 6.00 | 10.54 | 6.16 |
    | Multilingual phoneme small | 90 MB | 8.02 | 3.37 | 5.68 | 4.04 | 8.29 | 5.77 | 6.05 | 18.07 | 8.32 | 8.53 | 7.61 |
    | Multilingual phoneme medium | 218 MB | 6.70 | 2.63 | 4.53 | 3.12 | 5.95 | 3.95 | 4.61 | 14.81 | 6.04 | 8.47 | 6.08 |
    | Multilingual phoneme large | 543 MB | 5.42 | 1.96 | 3.52 | 2.25 | 4.06 | 2.64 | 2.97 | 11.33 | 4.04 | 5.97 | 4.43 |

* %WER with 4-gram LM 
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | 90 MB | 10.59 | 7.91 | 15.58 | 9.26 | 1.03 | 8.84 | 1.62 | 8.37 | 8.46 | 9.75 | 8.14 |
    | Multilingual bpe small | 92 MB | 12.00 | 9.82 | 12.40 | 9.98 | 3.29 | 9.67 | 3.31 | 9.95 | 9.11 | 13.56 | 9.30 |
    | Multilingual phoneme small | 90 MB | 10.76 | 8.68 | 16.01 | 9.98 | 1.02 | 7.32 | 1.59 | 6.714 | 7.63 | 7.30 | 7.64 |
    | Multilingual phoneme medium | 218 MB | 9.83 | 7.82 | 14.94 | 9.04 | 0.91 | 6.57 | 1.65 | 5.65 | 7.27 | 7.37 | 7.10 |
    | Multilingual phoneme large | 543 MB | 8.80 | 7.02 | 14.02 | 8.16 | 0.94 | 6.22 | 1.46 | 5.06 | 7.05 | 6.92 | 6.56 |


#### Crosslingual for Polish
* %PER
    | Model | 10 minutes | 1 hour | 10 hours | 130 hours |
    | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | 95.09 | 86.01 | 29.7 | 2.8 |
    | Phoneme PT and phoneme FT | 25.7 | 17.9 | 10.4 | 1.9 |

* %WER with 4-gram LM
    | Model | 10 minutes | 1 hour | 10 hours | 130 hours |
    | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | 100 | 99.98 | 13.86 | 4.97 |
    | Monolingual subword | - | 98.38 | 59.43 | 7.12 |
    | Phoneme PT and phoneme FT | 11.0 | 6.95 | 5.27 | 4.30 |
    | Phoneme PT and subword FT | 81.62 | 8.63 | 4.83 | 3.82 |
    | Subword PT and subword FT | 52.52 | 9.16 | 4.89 | 3.76 |

#### Crosslingual for Indonesian
* %PER
    | Model | 10 minutes | 1 hour | 10 hours | 130 hours |
    | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | - | 96.5 | 26.3 | 5.74 |
    | Phoneme PT and phoneme FT | 32.59 | 21.64 | 7.90 | 4.79 |

* %WER with 4-gram LM
    | Model | 10 minutes | 1 hour | 10 hours | 130 hours |
    | ------ | ------ | ------ | ------ | ------ |
    | Monolingual phoneme | - | - | 7.71 | 3.28 |
    | Monolingual subword | - | 96.42 | 49.67 | 10.85 |
    | Phoneme PT and phoneme FT | 6.85 | 3.27 | 2.54 | 2.43 |
    | Phoneme PT and subword FT | 98.65 | 24.57 | 3.59 | 2.92 |
    | Subword PT and subword FT | 87.75 | 23.56 | 3.91 | 3.07 |
