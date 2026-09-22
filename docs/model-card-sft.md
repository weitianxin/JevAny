---
library_name: peft
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-classification
tags:
  - decision-model
  - lora
  - calibration
---

# JevAny-27B-SFT v0.1.0

JevAny-27B-SFT is a rank-16 LoRA adapter and pointer head for `Qwen/Qwen3.8-27B`. It maps a shared text state and typed questions to option probabilities without decoding answer tokens.

## Training

- Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Adapter: rank 16 LoRA
- Pointer dimension: 256
- Training data: public classification, compositional, policy, and missing-option decisions
- Calibration: temperature fitted on a separate development partition

The base weights remain frozen. JevAny trains the adapter and pointer head together.

## Evaluation

On the held-out `transfer-v9` development panel: 81.36% knowable accuracy, 64.00% MMLU-Pro accuracy, 0.273 calibrated Brier score, and 51.63% coverage at no more than 5% empirical error. Median one-question H200 latency was 156.84 ms.

See the repository's [release results](https://github.com/weitianxin/JevAny/blob/main/results/release-v0.1.json) for the complete comparison and measurement notes.

## Limits

The release path is text-only. Its training envelope is 2,048 packed tokens; longer serving inputs were not trained as a first-class capability. Calibration can shift under new data. The adapter requires the separately distributed Qwen base weights and JevAny runtime.
