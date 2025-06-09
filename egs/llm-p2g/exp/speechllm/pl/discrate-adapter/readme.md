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
    test_pl  %SER 13.32 | %WER 5.20 [ 3094 / 59464, 473 ins, 322 del, 2299 sub ]
    ```
##### %WER with 4-gram LM
    ```
    test_pl  %SER 12.73 | %WER 5.05 [ 3003 / 59464, 352 ins, 387 del, 2264 sub ]
    ```



### Resource
| Config file | Checkpoint model | Tensorboard log |
| ----------- | ----------- | ----------- |
| [`config.json`](./config.json) | [`best.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/mt5_kmeans_pl_best.ckpt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064742186&Signature=dzQKOzNe%2Frd5IFw1O4DpiU79Hb8%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/tb_log_mt5_kmeans_pl.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064742214&Signature=B1Qi0iUGGfrpS0u75602qomGlk8%3D) |

