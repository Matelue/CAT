# Speechllm model with pooling adapter for German
Author: Ma, Te (mate153125@gmail.com)
### Basic info

This model is trained following the architecture of [Speechllm](https://github.com/skit-ai/SpeechLLM). The training dataset are the same as our proposed [LLM-P2G](../../../tkm/tkm.md) with TKM.

### Training and evaluation

* Follow the steps of [Speechllm](https://github.com/skit-ai/SpeechLLM#training) to train and test model. The configuration file [config.json](config.json) contains the experimental setup.

|    Monitor figure   |
|:-----------------------:|
|<div align="center"><img src="./monitor.png" width="400"/></div> |

### Results
##### %WER
    ```
    test_de  %SER 54.48 | %WER 13.65 [ 20253 / 148339, 2363 ins, 1692 del, 16198 sub ]
    ```
##### %WER with 4-gram LM
    ```
    test_de  %SER 53.60 | %WER 13.35 [ 19806 / 148339, 1960 ins, 1957 del, 15889 sub ]
    ```



### Resource
| Config file | Checkpoint model | Tensorboard log |
| ----------- | ----------- | ----------- |
| [`config.json`](./config.json) | [`best.ckpt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/mt5_pooling_de_best.ckpt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064741863&Signature=aRNxuEeekWiHxr2yJDD2Z%2FpfvoQ%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/tb_log_mt5_pooling_de.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064741917&Signature=KB5fd%2FvY8BhNyxNvAh12h6hxf6o%3D) |

