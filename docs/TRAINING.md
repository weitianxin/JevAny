# Train your own decision model

The trainer fits a LoRA adapter and pointer head over a frozen open backbone.
Training rows use the same `state` and `questions` as inference, plus labels.
The output directory can immediately be loaded by `JevModel` or `jevany serve`.

## Start with a small backbone

From the repository root, with Python 3.12 or newer:

```bash
python -m pip install -e '.[train]'
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

The starter has 24 original synthetic training tickets and 8 separate development
tickets, each covering `choice`, `noul`, and `score`. It teaches the workflow; it
is too small to train a general-purpose decision model.

The recipe uses `Qwen/Qwen3.5-0.8B`, BF16 CUDA weights, LoRA rank 16 and three epochs.
Replace `data` with your own labelled JSONL and choose a backbone that fits your
device. For a CPU experiment, override the CUDA settings:

```bash
jevany train --config recipes/sft.toml \
  --device cpu --dtype fp32 --weights-dtype fp32 --out runs/cpu-jev
```

CPU training is practical for small experiments, and can be slow. The full frozen
backbone, adapters, activations and optimizer memory must fit on each device.
The distributed implementation is DDP: adding devices increases throughput; it
does not split one backbone across GPUs.

## Recipes and overrides

Recipes are flat TOML files containing the trainer's argument names. Paths are
relative to the **current working directory**. Command-line flags override recipe
values. Unknown keys, invalid types, invalid labels and existing output directories
fail before weights load. `--dry-run` validates configuration and data; token
limits and backbone compatibility are checked when training starts.

| Recipe | Starting point | Output |
|---|---|---|
| [`sft.toml`](../recipes/sft.toml) | Open Qwen3.5-0.8B base | `runs/my-jev` |
| [`finetune.toml`](../recipes/finetune.toml) | Released JevAny-27B-SFT adapter | `runs/domain-jev` |
| [`rlcr.toml`](../recipes/rlcr.toml) | `runs/my-jev` from the small SFT recipe | `runs/my-jev-rlcr` |

Fine-tune the released checkpoint with your data:

```bash
python -m pip install -e '.[train,multimodal]'
jevany train --config recipes/finetune.toml \
  --data data/my-domain.jsonl --out runs/domain-jev
```

`init_from` restores the adapter and head into a new training run, with a new
optimizer and schedule. It is not an interrupted-job resume. The base revision,
LoRA rank and targets, head dimension, and multimodal settings must match.
The released checkpoint uses the native vision path even for text requests.

For Python applications:

```python
from jevany.training import train

checkpoint = train("recipes/sft.toml", output_dir="runs/python-jev")
```

All trainer arguments are available through `jevany train --help`
and `python -m jevany.train`. Both `--head_dim` and `--head-dim` spellings work.

## Data beyond the starter

The installable builders convert public sources into the same JSONL format:

```bash
jevany data build-sft --help
jevany data build-rlcr --help
```

The SFT builder covers HelpSteer3, Hermes tool decisions, ARC, QASC,
CommonsenseQA, ScienceQA, A-OKVQA, and VideoFeedback, with AI2D/MMMU reserved
for evaluation. It downloads upstream data and writes source revisions,
licenses, split counts and checksums in its manifest. Defaults include large
datasets and media; inspect its options before starting a build.

A text-only build with 500 examples from ARC/QASC/CommonsenseQA:

```bash
jevany data build-sft --out data/text \
  --helpsteer 0 --agent 0 --hard-text 500 \
  --scienceqa 0 --aokvqa 0 --video 0 --eval-per-source 0 \
  --tokenizer Qwen/Qwen3.5-0.8B
jevany train --config recipes/sft.toml \
  --data data/text/train.jsonl --out runs/text-jev
```

This command writes only a training split; bring a separate evaluation set.
Training from `--suite` additionally requires the base revision to be pinned
by the suite or by `--base-revision`. Full data fields, licenses and the
release mixtures are documented in [DATA.md](DATA.md).

## Multiple GPUs and hosts

Use one process per visible GPU on a workstation:

```bash
bash infra/train.sh --config recipes/sft.toml --out runs/distributed-jev
```

The same launcher works inside a scheduler allocation. On every host, set
`NNODES`, its distinct `NODE_RANK`, and `MASTER_ADDR` to the rank-0 host.
Mount the same input data, base weights and output directory on all hosts.
Keep the GPU visibility assigned by the scheduler.

The launcher defaults to all visible GPUs and prints the process topology.
Each worker reports its rank, world size, CUDA device and visible device mask.
When using fewer processes, set `PROCESSES_PER_HOST` and `GPU_SHORTFALL_REASON`.
Check these logs against the scheduler allocation; visibility alone does not
prove that every reserved GPU is active.

This launcher delegates allocation to your workstation or scheduler. It does
not create cloud resources. Select SFT or RLCR with the same
[`recipes`](#recipes-and-overrides) used for a single GPU.

## Evaluation and checkpoints

Training supports held-out suite evaluation, periodic checkpoints, early stopping,
and optional W&B logging. For example, add `--eval-suite data/eval-suite
--eval-every-steps 100 --checkpoint-every-steps 100` to a run. The evaluation suite
needs a manifest plus calibration and development splits; a plain JSONL file
is not a suite. Use `--wandb-project` only when you want W&B logging.

The final directory contains the adapter, tokenizer, `head.pt`,
`training_config.json`, and `training_metrics.json`. If early stopping selects a
different checkpoint, `selection.json` records that path; the root directory
always holds the final weights. Serve the selected path explicitly.

SFT is the default recipe. RLCR is an experimental continuation with calibration
rewards; its released version did not improve overall transfer accuracy.
Measure task accuracy and calibration on your own held-out data before selecting
a checkpoint. See [the objective](ALGORITHM.md) and [release results](EVALUATION.md).

## Backbone support

The implementation uses PyTorch, Transformers and PEFT. The text adapter loads
the base component of a Transformers causal language model, adds a pointer head,
and trains LoRA plus any newly added decision-token embeddings. The same trainer,
checkpoint format and serving API apply across model families.

The [model catalog](supported-models.json) pins the 26 selected official
checkpoints across seven families. It records each repository, revision, series,
size and native image/video support. Base, Thinking and quantized variants are
not additional entries in this catalog.

| Family | Selected models | Checkpoints |
|---|---|---:|
| Qwen | Qwen3.8-27B; Qwen3.6-27B; Qwen3.6-35B-A3B; Qwen3.5-0.8B; Qwen3.5-2B; Qwen3.5-4B; Qwen3.5-9B; Qwen3.5-27B; Qwen3.5-35B-A3B | 9 |
| Gemma | gemma-4-E4B-it; gemma-4-12B-it; gemma-4-31B-it | 3 |
| Muse | Muse-Glimmer-30B | 1 |
| Mistral | Ministral-3-3B-Instruct-2512-BF16; Ministral-3-8B-Instruct-2512-BF16; Ministral-3-14B-Instruct-2512-BF16; Devstral-Small-2-24B-Instruct-2512 | 4 |
| GLM | GLM-4.7-Flash; GLM-4.6V-Flash | 2 |
| Nemotron | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; NVIDIA-Nemotron-3-Nano-4B-BF16; NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 3 |
| Llama | Llama-3.2-1B-Instruct; Llama-3.2-3B-Instruct; Llama-3.2-11B-Vision-Instruct; Llama-3.1-8B-Instruct | 4 |

All families share `BackboneAdapter`, recipe flags, checkpoint loading and the
serving API. Choose a publisher repository or a local weights directory with
`--base`; there is no family-specific trainer. Native media selection reads
`config.json`, so local snapshots work without depending on repository names.
Gemma 4 Unified and GLM 4.6V use their native Transformers models and processors.

Every GPU holds the full base, including all MoE experts; active parameter counts
do not describe the required weight memory. The small starter remains
`Qwen/Qwen3.5-0.8B`. Gated repositories require the publisher's license acceptance
and Hugging Face access before downloading; local snapshots need no account.

Phi, Qwen 2.5/3/3-VL/3-Coder, Gemma 3/3n, Pixtral and older Mistral/Magistral releases are outside
the maintained scope. The Phi conversion utilities and legacy media adapter names
have been removed. The generic text adapter and custom adapter interface remain
available for other architectures, without a compatibility claim.

The released JevAny adapters identify `Qwen/Qwen3.8-27B` in their release manifest
and adapter configuration. Its Transformers architecture is `qwen3_5`.
Architecture compatibility does not make different base revisions interchangeable:
load each adapter with its recorded base and revision.

Select the base in the existing recipe or on the command line:

```bash
jevany train --config recipes/sft.toml \
  --base zai-org/GLM-4.7-Flash --out runs/glm-decisions
```

The optional adapter settings are:

```toml
backbone_adapter = "auto" # chooses the native media adapter when multimodal=true
branch_mode = "auto"      # packed where supported; independent causal rows otherwise
lora_targets = "all"     # supported linear modules, preserving the existing Qwen layout
# lora_target_modules = "q_proj,v_proj" # explicit names override the preset
```

`rows` repeats the shared state for each question. It preserves isolation without
requiring a custom attention mask and supports sliding-window and recurrent
backbones. `packed` requires an adapter that declares support for that mask;
unsupported combinations fail before training. Option isolation requires packed
mode. Prefix caching is enabled only for validated FP32 backbones. BF16/FP16
models and Qwen's `qwen3_5_text` recurrent architecture use full-forward inference:
real Qwen 27B and Llama 8B checks found excessive BF16 drift when reusing prefix
states. Serving selects full-forward inference automatically; unsupported direct
prefix-cache calls fail explicitly.

`attn` includes GLM's low-rank query/key/value projections, Nemotron's Mamba
input projection, and fused QKV projections. A fused QKV layer
cannot apply the `qv` preset to Q and V independently; use `attn`, `all`, or explicit
module names. Explicit names are checked against the loaded model, so a misspelled
target does not silently produce a partial adapter.

LoRA targets are resolved from actual `nn.Linear` and `Conv1D` modules. Fused
expert tensors and GLM/Nemotron routers stay frozen. Nemotron's Mamba `out_proj`
is excluded because its fused training kernel bypasses the module's forward
method. With PEFT 0.21, GLM's dense/shared MLP projection names are remapped to
fused expert parameters even when fully qualified; GLM therefore uses attention
LoRA for `all`, `dense`, and `attn`. Explicit requests for these incompatible
targets fail with an error. Neither model enables packed masks or prefix reuse.
Nemotron can use Transformers' PyTorch Mamba fallback without optional CUDA
kernels; large runs benefit from the upstream optimized kernels.

Non-Qwen tokenizers receive five `<|jev_*|>` tokens. Their embeddings are
initialized from the base embedding mean and trained automatically, including
when the base has spare vocabulary rows. Other embedding rows stay frozen.
Checkpoints save the tokenizer and learned token rows along with LoRA and the
pointer head. Loading restores that tokenizer before constructing the model.
Gemma E4B also has a frozen per-layer token table. New rows in that table use its
original mean embedding on both initial training and checkpoint reload.
Older Qwen checkpoints continue to use their original delimiters and metadata
defaults.

For another architecture, first try the text adapter. Unknown architectures use
independent rows by default. To customize loading, LoRA selection, packed-mask
support, cache construction, or native media processing, subclass
[`BackboneAdapter`](../jevany/backbones.py) in an installed Python module:

```python
from jevany.backbones import BackboneAdapter

class MyAdapter(BackboneAdapter):
    def supports_packed(self, config):
        return False
```

Set `backbone_adapter = "my_package.adapters:MyAdapter"` in the recipe. The import
path is saved in the checkpoint; install the same trusted adapter package in the
serving environment. Text backbones must accept token IDs, position IDs and an
attention mask and expose `last_hidden_state` and input embeddings. Native
media processing is handled by the vision adapters below.

### Native multimodal training

Set `multimodal = true` and choose a native vision base. The data format, training
loop, LoRA settings and checkpoint loading stay the same:

| Family | Vision base | Adapter | Media |
|---|---|---|---|
| Qwen | All nine Qwen 3.5, 3.6 and 3.8 catalog entries | `qwen_vl` | Images, video |
| Gemma | Gemma 4 E4B, 12B Unified, 31B IT | `gemma4_vision` | Images, video |
| Muse | `meta-models/Muse-Glimmer-30B` | `muse_vision` | Images, video |
| Mistral | Ministral 3 3B/8B/14B, Devstral Small 2 24B | `mistral_vision` | Images |
| Llama | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `llama_vision` | Images |
| GLM | `zai-org/GLM-4.6V-Flash` | `glm_vision` | Images, video |

`auto` selects from the base's configuration. A text-only variant does not gain
vision support by setting the flag. Unsupported media types fail with an error.
Audio is not part of the current training/request format.
Muse's native processor expands image/video placeholders and supplies frame
timestamps. Its adapter caps images at 512 merged tokens per record and videos
at eight sampled frames with a per-frame token budget. Processor metadata is used
during encoding and is not passed to the decoder.

For example, on a local CUDA GPU:

```bash
python -m pip install -e '.[train,multimodal]'
jevany train --config recipes/sft.toml --multimodal \
  --base google/gemma-4-31B-it --data data/images.jsonl --out runs/gemma-vision \
  --lr 0.00002 --head-lr 0.00002
```

Each media record contains one question. Native media records do not support
`option_isolation` or prefix-cache reuse. Relative media paths resolve against the
JSONL file; use the same labelled request format as text training:

```json
{"state":"Inspect the sample.","media":[{"type":"image","uri":"red.png"}],"questions":{"color":{"type":"choice","instructions":"What color is the sample?","criteria":{"red":"Red","blue":"Blue"},"label":"red"}}}
```

Training freezes the existing vision encoder and projector, and fits the language
decoder's LoRA, decision head and added decision-token embeddings. The native
processor handles image token expansion and model-specific inputs such as
Llama's cross-attention mask and Gemma's token types. Checkpoints save the complete
processor and selected adapter, so serving and subsequent training use the same
encoding. Text-only and image records can share a run; DDP accounts for Llama's
cross-attention parameters being unused on text-only records.

Devstral's official FP8 weights are dequantized for BF16 LoRA training, including
on A100 GPUs. Ministral's listed release already contains BF16 weights. Both use
the upstream tokenizer regex correction. The text adapter can extract a language
decoder from a composite vision base for text-only training.

For another native vision architecture, subclass `VisionAdapter` from
`jevany.backbones`. Set `media_types` and `language_model_path`; override
`process_media` or `forward_media` for another processor/model protocol.
`encode_media` can handle architectures needing different readout positions.
Set `conditional_parameters = True` if some trainable decoder paths are absent
on text-only records. Select the installed class through the existing
`backbone_adapter = "my_package.adapters:MyVisionAdapter"` setting. Models return
token hidden states for the shared decision head; the trainer does not contain
family-specific forward branches.

Both local training and `torchrun` use these interfaces without a cloud account
or resource-provider SDK.

### Short compatibility checks

The [pretrained validation record](../results/model-support-v1.json) includes all
26 current catalog checkpoints with 80 passing input/objective checks. The
original 37-checkpoint record is retained as historical evidence, including
models since removed from the maintained scope. It includes
revisions, learning rates, precision, reload errors, serving checks and failed
attempts. Most SFT checks use 12 steps, followed by two RLCR steps. The Gemma 4
12B video check uses 48 SFT steps, `--dtype fp32`, `--lr 2e-7` and
`--head-lr 1e-4`, with BF16 frozen weights.

The catalog contract test checks that every maintained checkpoint has matching
revision, modality, SFT, RLCR, reload and Python/HTTP serving evidence. Media
checks include mixed text records. These short runs establish compatibility;
they do not establish task quality, production throughput or every distributed
topology.

The offline tests use real, tiny Transformers architectures for the selected
families, plus a GPT-2 fixture for the generic custom-backbone contract. They
cover sliding-window and recurrent layers, MoE forwards, LoRA selection and gradients, added-token embeddings,
branch isolation, supported cache reuse, native media, checkpoint reload, and
SFT-to-RLCR continuation:

```bash
python -m pytest tests/test_model_support.py tests/test_backbones.py -q
python -m pytest tests/test_multimodal_backbones.py tests/test_serving.py tests/test_smoke_backbone.py -q
```

To check pretrained weights without completing a training run:

```bash
python -m pip install -e '.[dev]'
python -m scripts.smoke_backbone \
  --base Qwen/Qwen3.5-0.8B --steps 12 --out runs/smoke-qwen

TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
torchrun --standalone --nproc_per_node=2 --module scripts.smoke_backbone \
  --base Qwen/Qwen3.8-27B --steps 12 --lr 0.00002 --head-lr 0.00002 \
  --out runs/smoke-qwen-ddp
```

Use `--revision` to pin weights and `--device cpu` for a sufficiently small base.
Use `--base-load-path /path/to/snapshot` to load a predownloaded copy while keeping
the official `--base` and `--revision` in the checkpoint.
The smoke script exposes `--lr` and `--head-lr` for bases with different gradient
scales; its default head learning rate is `1e-4`.
Use `--dtype fp32` to disable training autocast while retaining BF16 frozen
weights on CUDA; the report records this separately from the checkpoint's weight dtype.
The script runs the actual trainer, measures uncalibrated NLL before and during
training, reloads the checkpoint, and checks predictions through both the Python
runtime and the FastAPI application. The HTTP checks exercise health, model
description, repeated typed requests, and rejection of invalid requests. It checks
prefix-cache parity where supported and explicit rejection elsewhere.
It writes `smoke.json`, per-rank GPU identity, and the trainer's evaluation
history. A passing SFT run requires finite adapter weights, updated LoRA weights,
lower final NLL, and successful prediction checks. Its small evaluation probe
intentionally reuses training examples; this is an optimization and compatibility
test, not an accuracy benchmark. The probe covers all eight training rows,
including both media and text rows when `--mixed-text` is enabled. Media probes
score each row in both choice orders, matching the trainer's option permutation.

To check RLCR continuation and serving from the resulting checkpoint:

```bash
python -m scripts.smoke_backbone --base Qwen/Qwen3.5-0.8B \
  --init-from runs/smoke-qwen/checkpoint --rlcr --steps 2 --out runs/smoke-qwen-rlcr
```

RLCR must update the saved LoRA tensors and retain finite losses and predictions.
Its combined reward objective does not require the supervised NLL to decrease.

For native media, the same smoke script generates its own image or video files:

```bash
python -m scripts.smoke_backbone --base Qwen/Qwen3.8-27B \
  --media image --mixed-text --steps 12 --lr 0.00002 --head-lr 0.00002 \
  --out runs/smoke-qwen-vision
```

Use `--media video` for a video-capable base, or run the same module under
`torchrun`. The media check also requires predictions to change when the media
changes while the question and state stay fixed.

For PyTorch 2.6 / CUDA 12.6, use `TORCH_NCCL_USE_COMM_NONBLOCKING=0`.
[`infra/train.sh`](../infra/train.sh) sets this default before starting workers
and preserves an explicit override.
PyTorch 2.6 uses DDP's standard initial parameter synchronization; newer versions
with `init_sync` support can avoid broadcasting the frozen base.

The runtime admits up to 2,048 packed training tokens by default. Multimodal rows
contain one question and require a native vision-capable base and
`multimodal = true`. Model weights and third-party data keep their upstream
licenses. The complete frozen suites and raw media from the release are not
included, so the default builder alone does not reproduce the published run.
