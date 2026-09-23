---
library_name: peft
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - decision-model
  - lora
  - calibration
  - agents
---

# JevAny-27B-SFT v0.2

JevAny-27B-SFT is the recommended general JevAny checkpoint. It is a rank 16 LoRA adapter plus a pointer head for Qwen/Qwen3.8-27B. It maps shared state and typed questions to option probabilities without decoding answer tokens.

## Intended Use

Use this checkpoint for bounded classification, routing, ranking, tool choice, agent actions, and confidence-aware decisions. It accepts choice, noul, and ordinal score questions through the JevAny API. It is not a chat or chain-of-thought model.

## Training

The adapter and pointer head were trained on 107,278 records spanning preferences, agent and tool decisions, hard and many-choice reasoning, classification, policy, and native image and video evidence. A separate calibration partition sets the inference temperature. The base weights remain frozen.

## Evaluation

| Evaluation | Result |
|---|---:|
| v2 development accuracy | 90.34% |
| v2 development NLL | 0.265 |
| transfer-v9 accuracy | 82.41% |
| MMLU-Pro accuracy | 73.0% |
| AI2D accuracy | 86.0% |
| MMMU accuracy | 68.0% |

Development metrics exclude a 100-question single-class VideoFeedback slice. It exercises the native video path but is not a meaningful capability benchmark. AI2D and MMMU use native images through the backbone's vision path. Full measurements and the Jev comparison are in the repository [release results](https://github.com/weitianxin/JevAny/blob/main/results/release-v0.2.json).

## Limits

The released path accepts text, JSON-renderable state, native images, and native video. Multimodal requests currently contain one isolated question and use a bounded visual token budget. HTTP media is operator opt-in through a controlled local root; network media URLs are rejected. The trained text context envelope is 2,048 packed tokens. Calibration can shift under new data, so validate thresholds on a deployment-specific calibration set. The adapter requires the separately distributed Qwen base weights and JevAny runtime.
