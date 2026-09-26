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

The recipe uses `Qwen/Qwen2.5-0.5B`, BF16 CUDA weights, LoRA rank 16 and three epochs.
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
| [`sft.toml`](../recipes/sft.toml) | Open Qwen2.5-0.5B base | `runs/my-jev` |
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

All original research arguments remain available through `jevany train --help`
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
  --tokenizer Qwen/Qwen2.5-0.5B
jevany train --config recipes/sft.toml \
  --data data/text/train.jsonl --out runs/text-jev
```

This command writes only a training split; bring a separate evaluation set.
Training from `--suite` additionally requires the base revision to be pinned
by the suite or by `--base-revision`. Full data fields, licenses and the historical
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
not create cloud resources. Historical release commands remain in
[`train_sft.sh`](../scripts/train_sft.sh) and
[`train_rlcr.sh`](../scripts/train_rlcr.sh); they require the original frozen suites.

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

| Family | Example base | Adaptation |
|---|---|---|
| Qwen | `Qwen/Qwen3-0.6B` | Reuses existing delimiters where available and preserves the Qwen LoRA layout |
| Llama | `meta-llama/Llama-3.2-1B` | Adds five decision tokens; uses the base model's input embedding interface |
| Gemma | `google/gemma-3-1b-pt` | Adds decision tokens; uses independent rows for sliding-window attention |
| Mistral | `mistralai/Mistral-7B-v0.3` | Supports full attention and sliding-window configurations |
| Phi | `microsoft/Phi-4-mini-instruct` | Supports fused QKV and MLP projections through `all` or `attn` LoRA targets |

These are representative text backbones, not a claim about every model carrying
the same family name. Some upstream weights require accepting their license and
authenticating with Hugging Face. The starter remains Qwen2.5; released adapters
still use their original Qwen base and cannot be moved onto another backbone.

Select the base in the existing recipe or on the command line:

```bash
jevany train --config recipes/sft.toml \
  --base meta-llama/Llama-3.2-1B --out runs/llama-decisions
```

The optional adapter settings are:

```toml
backbone_adapter = "auto" # text by default; qwen_vl with multimodal=true
branch_mode = "auto"      # packed where supported; independent causal rows otherwise
lora_targets = "all"     # all linear layers, preserving the existing Qwen layout
# lora_target_modules = "q_proj,v_proj" # explicit names override the preset
```

`rows` repeats the shared state for each question. It preserves isolation without
requiring a custom attention mask and supports sliding-window and recurrent
backbones. `packed` requires an adapter that declares support for that mask;
unsupported combinations fail before training. Option isolation requires packed
mode. Prefix caching is checked against uncached inference in the backbone tests.

`attn` includes fused projections such as Phi's `qkv_proj`. A fused QKV layer
cannot apply the `qv` preset to Q and V independently; use `attn`, `all`, or explicit
module names. Explicit names are checked against the loaded model, so a misspelled
target does not silently produce a partial adapter.

Non-Qwen tokenizers receive five `<|jev_*|>` tokens. Their embeddings are
initialized from the base embedding mean and trained automatically, including
when the base has spare vocabulary rows. Other embedding rows stay frozen.
Checkpoints save the tokenizer and learned token rows along with LoRA and the
pointer head. Loading restores that tokenizer before constructing the model.
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
image/video processing remains a separate adapter: `qwen_vl` preserves the
existing Qwen implementation, and other vision architectures require their own
media adapter.

### Short compatibility checks

The offline tests use real, tiny Transformers architectures for all five
families, including sliding-window attention, fused projections, added-token
gradients, branch isolation, cache reuse, checkpoint reload, and a custom adapter's
SFT-to-RLCR continuation:

```bash
python -m pytest tests/test_backbones.py -q
```

To check pretrained weights without completing a training run:

```bash
python -m scripts.smoke_backbone \
  --base Qwen/Qwen3-0.6B --steps 12 --out runs/smoke-qwen

TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
torchrun --standalone --nproc_per_node=2 --module scripts.smoke_backbone \
  --base meta-llama/Llama-3.2-1B --steps 12 --out runs/smoke-llama-ddp
```

Use `--revision` to pin weights and `--device cpu` for a sufficiently small base.
The smoke script exposes `--lr` and `--head-lr` for bases with different gradient
scales; its default head learning rate is `1e-4`.
The script runs the actual trainer, measures uncalibrated NLL before and during
training, reloads the checkpoint, and checks predictions and prefix-cache results.
It writes `smoke.json`, per-rank GPU identity, and the trainer's evaluation
history. A passing run requires finite adapter weights, updated LoRA weights,
lower final NLL, and successful prediction checks. Its small evaluation probe
intentionally reuses training examples; this is an optimization and compatibility
test, not an accuracy benchmark.

The [recorded GPU checks](../results/backbone-smoke-v1.json) cover six pretrained
bases across the five families and four additional two-GPU DDP runs, each with
12 optimizer steps. The report pins the weight revisions, identifies the public
Llama/Gemma mirrors, and includes all probe losses and reload/cache differences.

The PyTorch 2.6 / CUDA 12.6 container used for the GPU checks required explicit
`TORCH_NCCL_USE_COMM_NONBLOCKING=0`: a standalone collective probe returned
incorrect gather/reduce values with the variable unset, and correct values plus
averaged gradients with it set to `0`. [`infra/train.sh`](../infra/train.sh) sets
this default before starting workers and preserves an explicit override.
PyTorch 2.6 uses DDP's standard initial parameter synchronization; newer versions
with `init_sync` support can avoid broadcasting the frozen base.

The runtime admits up to 2,048 packed training tokens by default. Multimodal rows
contain one question and require a native vision-capable base and
`multimodal = true`. Model weights and third-party data keep their upstream
licenses. The complete frozen suites and raw media from the release are not
included, so the default builder alone does not reproduce the published run.
