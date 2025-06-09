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
    test_de  %SER 60.11 | %WER 16.95 [ 25140 / 148339, 3014 ins, 2218 del, 19908 sub ]
    ```
##### %WER with 4-gram LM
    ```
    test_de  %SER 59.69 | %WER 16.73 [ 24812 / 148339, 2539 ins, 2573 del, 19700 sub ]
    ```



### Resource
| Config file | Checkpoint model | Tensorboard log |
| ----------- | ----------- | ----------- |
| [`config.json`](./config.json) | [`best.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/mt5_kmeans_de_best.ckpt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064742052&Signature=2cAiab57kiZPgiDSxTFG8RgVPAM%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/speechllm/tb_log_mt5_kmeans_de.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064742074&Signature=eHkv8f8n4RsOXQWofmC39aLCC1Y%3D) |

