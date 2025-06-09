# Speechllm method for intergrating ASR and LLM (mT5)
SpeechLLM integrates ASR and LLM into a single end-to-end framework, directly generating text from speech using a large language model. It typically employs a lightweight adapter module to bridge the ASR output and the LLM input. The adapter can use either continuous representations or discrete representations, enabling effective integration without retraining the LLM. 

## Training
This experiment uses the [Speechllm](https://github.com/skit-ai/SpeechLLM) toolkit. For fair comparison with our LLM-P2G, we frozen the acoustic model (Whistle-S2P) and fine-tune the LLM (mT5-base) with the 130 hours speech data. Two different adapters (continuous and discrete) are performed for Polish and German.

# Results
### WER of speechllm models
| Speechllm model | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| Whistle + pooling-adapter + mT5 | [4.31](../speechllm/pl/pooling-adapter/readme.md) | [4.15](../danp/pl/pooling-adapter/readme.md) | [13.65](../danp/de/pooling-adapter/readme.md) | [13.35](../danp/de/pooling-adapter/readme.md) |
| Whistle + discrate-adapter + mT5 | [5.25](../danp/pl/discrate-adapter/readme.md) | [5.05](../danp/pl/discrate-adapter/readme.md) | [16.95](../danp/de/discrate-adapter/readme.md) | [16.73](../danp/de/discrate-adapter/readme.md) |