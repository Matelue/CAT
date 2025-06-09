# Speechllm model with pooling adapter for Polish
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
    test_pl  %SER 12.05 | %WER 4.31 [ 2562 / 59464, 359 ins, 287 del, 1916 sub ]
    ```
##### %WER with 4-gram LM
    ```
    test_pl  %SER 11.30 | %WER 4.15 [ 2467 / 59464, 292 ins, 343 del, 1832 sub ]
    ```



### Resource
| Config file | Checkpoint model | Tensorboard log |
| ----------- | ----------- | ----------- |
| [`config.json`](./config.json) | [`best.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/mt5_pooling_pl_best.ckpt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064738127&Signature=aRzdhazPAqicl9kA%2FrE5xYOsxUY%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/tb_log_mt5_pooling_pl.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064742150&Signature=bCz7b%2F3%2FeNQCFuAbLnhbBHC%2BD5A%3D) |

