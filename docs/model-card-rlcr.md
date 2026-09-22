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

- Parent: JevAny-27B-SFT v0.1.0, step 786
- Selected checkpoint: optimizer step 384 of a planned 512-step epoch
- Hardware: 16 H200 GPUs across two nodes
- Data: 8,192 records; see the repository's [data documentation](https://github.com/weitianxin/JevAny/blob/main/docs/DATA.md)
- Effective batch: 16 source records
- LoRA learning rate: `5e-6`; pointer-head learning rate: `1e-5`
- RLCR group size: 32
- Exploration sigma: linear decay from `0.4` to `0.1`
- Supervised cross-entropy weight: `0.25`
- Temperature: `1.319507910772894`, fitted on 1,264 held-out development questions

The formal RLCR run took 14 minutes 28 seconds. Evaluation ran every 128 optimizer steps; step 384 was selected before the end of the epoch.

## Evaluation

On the held-out `transfer-v9` development panel: 81.84% knowable accuracy, 66.00% MMLU-Pro accuracy, 0.269 calibrated Brier score, and 54.11% coverage at no more than 5% empirical error. Mean confidence on explicitly unknowable questions was 0.428, with none at or above 0.9. Median one-question H200 latency was 153.75 ms.

Against SFT, accuracy changed by +0.48 percentage points (exact McNemar `p=0.332`). The supported conclusion is improved calibration and selective prediction, not a statistically established accuracy gain.

## Limits

This is not a generated-reasoning model and does not reproduce the RLCR paper's reasoning rollouts. The release path is text-only. Its training envelope is 2,048 packed tokens; longer serving inputs were not trained as a first-class capability. Confidence requires deployment-specific validation.
