---
library_name: peft
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-classification
tags:
  - decision-model
  - lora
  - calibration
  - reinforcement-learning
---

# JevAny-27B-RLCR v0.1.0

JevAny-27B-RLCR continues the released SFT checkpoint with decision-only reinforcement learning with calibration rewards. It is the recommended v0.1 checkpoint.

## Training

- Parent: JevAny-27B-SFT v0.1.0
- Data: hard examples, broad replay, compositional decisions, policy decisions, and knowable/unknowable pairs
- Objective: group-relative RLCR with a supervised cross-entropy anchor
- Calibration: temperature fitted on a separate development partition

The detailed data design is documented in the repository's [data guide](https://github.com/weitianxin/JevAny/blob/main/docs/DATA.md).

## Evaluation

On the held-out `transfer-v9` development panel: 81.84% knowable accuracy, 66.00% MMLU-Pro accuracy, 0.269 calibrated Brier score, and 54.11% coverage at no more than 5% empirical error. Mean confidence on explicitly unknowable questions was 0.428, with none at or above 0.9. Median one-question H200 latency was 153.75 ms.

Against SFT, accuracy changed by +0.48 percentage points (exact McNemar `p=0.332`). The supported conclusion is improved calibration and selective prediction, not a statistically established accuracy gain.

## Limits

This is not a generated-reasoning model and does not reproduce the RLCR paper's reasoning rollouts. The release path is text-only. Its training envelope is 2,048 packed tokens; longer serving inputs were not trained as a first-class capability. Confidence requires deployment-specific validation.
