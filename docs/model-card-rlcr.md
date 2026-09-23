---
library_name: peft
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - decision-model
  - lora
  - calibration
  - reinforcement-learning
---

# JevAny-27B-RLCR v0.2

JevAny-27B-RLCR is an experimental continuation of JevAny-27B-SFT using reinforcement learning with calibration rewards. It retains the same rank 16 LoRA and pointer-head architecture.

## Intended Use

Use this checkpoint to study calibration-reward training for bounded decisions. JevAny-27B-SFT remains the recommended general checkpoint because RLCR did not improve overall development or transfer accuracy.

## Training

The 40,000-record RL mixture emphasizes hard reasoning, many-choice questions, agent actions, preferences, mathematical and medical decisions, and image-derived and video-derived cases. The objective combines group-relative calibration reward with a supervised anchor. It is decision-only RL, not token-level GRPO, and it generates neither reasoning traces nor confidence tokens.

## Evaluation

| Evaluation | Result |
|---|---:|
| v2 development accuracy | 89.74% |
| v2 development NLL | 0.260 |
| transfer-v9 accuracy | 82.31% |
| MMLU-Pro accuracy | 73.5% |
| AI2D accuracy | 87.0% |
| MMMU accuracy | 63.0% |

Development metrics exclude a 100-question single-class VideoFeedback slice. Against SFT, transfer accuracy changed by -0.10 percentage points, with 3 fixes and 4 regressions. The paired 95% bootstrap interval is [-0.58, 0.39] points. Full measurements are in the repository [release results](https://github.com/weitianxin/JevAny/blob/main/results/release-v0.2.json).

## Limits

The small development NLL gain did not transfer consistently after independent calibration. Native multimodal requests currently contain one isolated question and use a bounded visual token budget. HTTP media is operator opt-in through a controlled local root; network media URLs are rejected. Confidence requires deployment-specific validation. The adapter requires the separately distributed Qwen base weights and JevAny runtime.
