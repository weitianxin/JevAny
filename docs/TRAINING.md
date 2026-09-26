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

The implementation uses PyTorch, Transformers and PEFT. It currently assumes
Qwen decision delimiters, Qwen-compatible projection names, and a supported
Transformers text/vision backbone interface. The starter uses Qwen2.5; the
released adapters use Qwen3.8-27B. Other architectures need an explicit adapter
and validation before being supported.

The runtime admits up to 2,048 packed training tokens by default. Multimodal rows
contain one question and require a native vision-capable base and
`multimodal = true`. Model weights and third-party data keep their upstream
licenses. The complete frozen suites and raw media from the release are not
included, so the default builder alone does not reproduce the published run.
