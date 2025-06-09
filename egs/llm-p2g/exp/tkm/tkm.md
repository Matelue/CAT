# Top-K marginalized (TKM) method for LLM-P2G
Beam search decoding and sanmpling are also uesd to generate noisy phonemes for marginalized training and decoding. Specially, a key feature of TKM is that the generation of these noisy phoneme sequences is performed on-the-fly during training, rather than through offline preprocessing. We also fine-tune the LLM-P2G (mT5-base) with different TKM strategies. 


## TKM strageties
| Strategy | Phoneme data for marginalized |
| ------ | ------ |
| top-8-beam |  Top-8 phoneme beam results|
| top-32-beam | Top-32 phoneme beam results|
| random-8-of-32-beam | random 8 of 32 phoneme beam results|
| random-8-sample-T | 8 Sampling phoneme results from logits with high temperature(T=1.5)|



## Results
### WER of LLM-P2G with TKM

| TKM strategy | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| random-8-of-32-beam_20h | [19.19](../tkm/pl/random-8-of-32-beam_20h/readme.md) | [17.36](../tkm/pl/random-8-of-32-beam_20h/readme.md) | [29.20](../tkm/de/random-8-of-32-beam_20h/readme.md) | [28.78](../tkm/de/random-8-of-32-beam_20h/readme.md) |
| top-32-beam | [16.55](../tkm/pl/top-32-beam/readme.md) | [16.12](../tkm/pl/top-32-beam/readme.md) | [21.69](../tkm/de/top-32-beam/readme.md) | [21.31](../tkm/de/top-32-beam/readme.md) |
| top-8-beam | [4.31](../tkm/pl/top-8-beam/readme.md) | [3.80](../tkm/pl/top-8-beam/readme.md) | [13.58](../tkm/de/top-8-beam/readme.md) | [13.18](../tkm/de/top-8-beam/readme.md) |
| random-8-of-32-beam | [4.01](../tkm/pl/random-8-of-32-beam/readme.md) | [3.68](../tkm/pl/random-8-of-32-beam/readme.md) | [13.44](../tkm/de/random-8-of-32-beam/readme.md) | [13.03](../tkm/de/random-8-of-32-beam/readme.md) |
| random-8-sample-T | [3.98](../tkm/pl/random-8-sample-T/readme.md) | [3.61](../tkm/pl/random-8-sample-T/readme.md) | [13.21](../tkm/de/random-8-sample-T/readme.md) | [12.94](../tkm/de/random-8-sample-T/readme.md) |


