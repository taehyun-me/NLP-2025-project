---
base_model: google/gemma-3-1b-it
library_name: peft
model_name: reward_model_output_2
tags:
- base_model:adapter:google/gemma-3-1b-it
- lora
- reward-trainer
- transformers
- trl
licence: license
---

# Model Card for reward_model_output_2

This model is a fine-tuned version of [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

text = "The capital of France is Paris."
rewarder = pipeline(model="None", device="cuda")
output = rewarder(text)[0]
print(output["score"])
```

## Training procedure

 


This model was trained with Reward.

### Framework versions

- PEFT 0.18.0
- TRL: 0.25.1
- Transformers: 4.57.3
- Pytorch: 2.8.0+cu129
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```