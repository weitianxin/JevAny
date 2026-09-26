# Model family comparison

This experiment is prepared but has not started training. It requires the
original v0.2 frozen data and benchmark assets, which are not distributed with
this repository. There are no new trained checkpoints or comparison scores yet.

## Models

Select one recent official checkpoint near 27B from each maintained non-Qwen
family. The catalog has no Llama checkpoint near 27B, so use its largest
supported release, 11B Vision. Produce an SFT adapter and an RLCR continuation
for each base. Reevaluate the two released Qwen adapters as baselines without
retraining Qwen.

| Family | Base | Native media used | Total size |
|---|---|---|---|
| Gemma | `google/gemma-4-31B-it` | Image, video | 31B |
| Muse | `meta-models/Muse-Glimmer-30B` | Image, video | 30B |
| Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512` | Image | 24B |
| GLM | `zai-org/GLM-4.7-Flash` | None | 30B MoE |
| Nemotron | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | None | 30B MoE |
| Llama | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Image | 11B |

Use the revisions in [the catalog](../supported-models.json). Both MoE bases
activate approximately 3B parameters, but each GPU loads all experts. This is
a comparison of usable model families, not an equal-parameter architecture
ablation. The newer Nemotron Lightning is chosen over Nano; recency and
popularity do not necessarily identify the same checkpoint.

## Training protocol

The [SFT recipe](../../recipes/family-sft.toml) and
[RLCR recipe](../../recipes/family-rlcr.toml) recover the objective and schedule
settings stored in the published adapters' `head.pt`. Their source revisions are:

- SFT: `tianxinwei/JevAny-27B-SFT@ad7b48b7056a9742f54aacc9b98b6b46dc2ce167`.
- RLCR: `tianxinwei/JevAny-27B-RLCR@078883b2edbab687585bc86fe5508059756c7f09`.

Use eight DDP workers per family, one full model per GPU. Batch size and
accumulation are both one, giving an effective batch of eight source records.
Run families independently in parallel; within each family, finish SFT before
starting RLCR.

| Setting | SFT | RLCR |
|---|---:|---:|
| Epochs | 2 | 1 |
| LoRA / head learning rate | 5e-5 | 2e-6 |
| LoRA rank / head dimension | 16 / 256 | 16 / 256 |
| Weight decay | 0.01 | 0.01 |
| Frozen weights / autocast | BF16 / BF16 | BF16 / BF16 |
| Evaluation / save interval | 1,000 steps | 200 steps |
| Seed | 0 | 0 |

SFT retains the release's none/distractor augmentations and paired examples.
RLCR disables them and uses group size 32, exploration sigma 0.4 to 0.2,
policy weight 0.25, and supervised CE weight 0.5. Both use gradient
checkpointing, a 1,024-token state limit, and a 2,048-token branch/packed limit.

Use independent causal rows across families. The exact trainable modules still
depend on the adapter: GLM uses attention LoRA, fused expert tensors stay frozen,
and non-Qwen decision token embeddings are trained. Record resolved targets,
trainable parameter counts, precision, and any stability changes. Do not call
those differences weight-matched training.

Run the complete epoch budgets and select the lowest calibrated development
NLL among saved checkpoints, excluding the single-class VideoFeedback slice.
Break ties by the earlier step. Fit temperature only on calibration records.
RLCR starts from that selected SFT's uncalibrated training weights. Transfer and
benchmark scores do not select checkpoints or learning rates.

The release selected SFT step 13,000 and RLCR step 1,000. These are provenance,
not stopping steps for different datasets or architectures. A short
compatibility check is not a completed training run.

## Data required before submission

The release records identify these original artifacts:

| Artifact | Manifest SHA-256 / identifier |
|---|---|
| SFT, 107,278 records | `5a9cdc1b15b470fd15032aeb6e8dbf7403ffbd4abd574577e76245a565fc1ec2` |
| RLCR, 40,000 records | `36ea433075fd85a770a50f989574e925624c50abc48a6fb00715e4d87bae02aa` |
| Transfer, 1,046 questions | `transfer-v9`; original manifest needed |
| MMStar, 1,330 questions | `5a0e350d5ec7af663bc70f6a0c3d06118fb314e756d35a6eb10efc4d8068e957` |
| MVBench, 600 questions | `eac1c0435e29a06b5ef62880f314700e6494b708282c60ad4d929c24c8a7e2de` |

Verify the manifests, partitions and referenced media before creating derived
views. Preserve original record IDs and split membership. Keep the same text
records for every family and retain media records only where the base supports
every required modality. Do not remove media from a visual question and treat
the result as an equivalent text example. Do not duplicate text to compensate
for excluded media.

Before tokenizer admission, the expected SFT view sizes are 107,278 for
image/video, 87,708 for image, and 65,496 for text. RLCR sizes are 40,000,
38,000 and 34,000 respectively. Thus text exposure and epoch counts match;
total records, optimizer steps and compute differ. Record these differences.
Admission must check every family tokenizer and processor, including SFT
augmentations; explicitly report exclusions and keep the common text panel
identical across families.

The public builders alone do not reconstruct the original core data or
`transfer-v9`. Rebuilding data would require a separately identified experiment
and fresh Qwen evaluation on its new panels. Do not attach the old scores to
newly sampled benchmark subsets.

## Quality and latency

Evaluate SFT and RLCR separately. Report accuracy, NLL, Brier, ECE, evaluated
counts, and rejected/truncated counts. Use paired question or media-group
bootstrap intervals for differences.

- Common text: development text, transfer-v9, MMLU-Pro and MuSR.
- Image-capable models: identical AI2D, MMMU and clean MMStar panels.
- Video-capable models: the three-task MVBench panel.
- Agent decisions: RAGEN FrozenLake and Sokoban, 50 episodes each, seeds
  beginning at 1,000 and 2,000, respectively, at most 64 actions. Pin RAGEN to
  `d97bb3284e99568adfd44ee15c736d7685c07512`.

Unsupported modalities are N/A. Keep the existing MMStar/MVBench full, blank
and shuffled controls. Compare only identical panel IDs with the same media
assets and option order. If a model cannot encode a record, resolve admission
before comparing; do not silently shrink its denominator.

Use the same GPU type and loading settings for all latency measurements,
including the published Qwen baselines. Run one serial inference process per
GPU, without overlapping training on that GPU. Report the GPU topology and
whether other GPUs on the host were active.

```bash
JEVANY_MERGE=0 python -m scripts.benchmark_latency \
  --run runs/gemma-31b-sft-selected \
  --suite data/common-text-evaluation --modality text \
  --records 256 --warmup 16 --repeats 3 --seed 0 \
  --out results-local/gemma-sft-text-latency.json
```

Repeat on the same image/video panels where supported. This script reports
median and p95 forward latency, local end-to-end latency including preprocessing,
input token counts, and peak allocated CUDA memory. It excludes model loading,
warmup and network time. Its serial requests/second is not saturated serving
throughput. Preserve its raw samples and panel hashes alongside quality results.
No latency ranking is available until those measurements run.
