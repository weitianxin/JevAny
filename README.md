<p align="center">
  <img src="docs/title.png" alt="JevAny: Your Jev from Any Model to Any Application" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API docs" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="Examples" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="Tests" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

Train and serve Jev-style decision models with JevAny: fine-tune an open language model or use our pretrained checkpoint. The shared [Python and HTTP APIs](docs/API.md) follow the Jev format, taking a state, question, and candidate answers and returning a choice with per-option probabilities.

<p align="center">
  <img src="docs/hero.png" alt="JevAny workflow: train a Jev model with multi-modal data and RLCR/SFT, then deploy through a unified API with test environments and practical examples" width="100%">
</p>

| Start here | What JevAny provides |
|---|---|
| **[Train a model](#train-a-model)** | Starter data, public-source data builders, SFT/RLCR recipes, and a multi-GPU training launcher |
| **[Deploy a checkpoint](#deploy-a-checkpoint)** | Released checkpoints, local Python inference, and an HTTP server |

## See it act

Examples built with [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT):

[![JevAny choosing actions across robotics, browser, software, laboratory and mobility tasks](docs/demos/jevany-cases.gif)](docs/CASES.md)

[Explore all 30 cases](docs/CASES.md), with task descriptions and decision records. These are selected successful runs, not a success-rate benchmark. Three scenarios also have [small runnable applications](#run-the-examples) using the public interface.

## Get the code

Python 3.12 or newer is required. Install the extras for the path you choose below.

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

## Deploy a checkpoint

The released 27B checkpoint needs a GPU that holds the full BF16 base model plus runtime memory; the showcase used A100 80 GB GPUs. First loading downloads the adapter and its separately distributed base weights. Smaller checkpoints you train use the same interface.

```bash
python -m pip install -e '.[serve,multimodal]'

jevany serve --checkpoint tianxinwei/JevAny-27B-SFT \
  --device cuda --dtype bf16 --port 8008
```

With the server running, make a decision from Python:

```python
from jevany import Choice, JevClient, Noul

jev = JevClient("http://127.0.0.1:8008")
result = jev.system_one(
    state={"ticket": "I was charged twice. Please help."},
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
        ),
        "urgent": Noul(instructions="Does this require urgent review?"),
    },
)
print(result["answers"]["department"]["choice"])
print(result["answers"]["department"]["probabilities"])
```

For inference inside your application, load once with `JevModel.from_pretrained(...)` and call the same `system_one` method:

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
```

You can also send JSON to `POST /v1/systemone`, run `jevany decide examples/request.json`, or point the official TypeSafe SDK at the server. [The deployment guide](docs/DEPLOYMENT.md) covers each option, hardware, offline loading and media inputs. A client-only installation (`pip install -e .`) does not install PyTorch.

## Train a model

Train Qwen, Llama, Gemma, Mistral or Phi through the same interface. Start with the included data and a small Qwen backbone:

```bash
python -m pip install -e '.[train]'

jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml

# Serve the checkpoint you just trained.
python -m pip install -e '.[serve]'
jevany serve --checkpoint runs/my-jev --model-name my-jev
```

The starter contains 24 original synthetic training records and 8 separate development records. Each asks choice, binary and score questions. It is for learning the workflow; train on representative domain data to build a useful model.

The SFT recipe uses `Qwen/Qwen2.5-0.5B` and a CUDA GPU. Change `base` and `data` in the TOML file, or override them on the command line. Labels use the inference format with a `label` added to each question. The trainer fits a LoRA adapter, pointer head and any added decision-token embeddings; the original base weights stay frozen.

| Next step | Command or guide |
|---|---|
| Train on your data | `jevany train --config recipes/sft.toml --data data/my-domain.jsonl --out runs/domain-jev` |
| Use another backbone | `jevany train --config recipes/sft.toml --base microsoft/Phi-4-mini-instruct --out runs/phi-jev` |
| Fine-tune our released checkpoint | [`recipes/finetune.toml`](recipes/finetune.toml) |
| Build larger datasets | `jevany data build-sft --help` · [sources and formats](docs/DATA.md) |
| Run on multiple GPUs or hosts | [`infra/train.sh`](infra/train.sh) |
| Experiment with calibration rewards | [`recipes/rlcr.toml`](recipes/rlcr.toml) |

[The training guide](docs/TRAINING.md) covers backbone adapters, CPU overrides, Python training, evaluation and distributed launch. DDP keeps a full model on each GPU.

Six pretrained bases across these five families passed 12-step GPU training with lower final loss and successful checkpoint reloads. See the [compatibility test results](results/backbone-smoke-v1.json).

## Run the examples

With a server running:

```bash
python -m examples.inbox
python -m examples.sql_repair
python -m examples.service_recovery
```

| Example | What you can build from it |
|---|---|
| [Inbox triage](examples/inbox.py) | Classify messages and decide which need a reply |
| [SQL repair](examples/sql_repair.py) | Select a query, execute it in SQLite, and check its result |
| [Service recovery](examples/service_recovery.py) | Run a multi-step agent against a local replica simulator |

Pass `--checkpoint runs/my-jev` to run any example in-process. The SQL and recovery examples report failed checks as failures. [Example instructions](examples/README.md) explain the environments; [harness and symbolic integrations](docs/INTEGRATIONS.md) add optional LLM planning.

## Checkpoints and evidence

| Checkpoint | Role | Transfer accuracy |
|---|---|---:|
| [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) | Default released model | 82.41% |
| [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) | Experimental RLCR continuation | 82.31% |

These v0.2 measurements cover 1,046 transfer questions. RLCR has not shown an overall transfer gain. [Evaluation details](docs/EVALUATION.md) include the full comparison, image/video controls and negative test-time adaptation results; [ALGORITHM.md](docs/ALGORITHM.md) describes the training objective.

Native images and video require a compatible vision checkpoint and one question per request. HTTP media inputs are enabled explicitly through `JEVANY_MEDIA_ROOT`. The validated training window is 2,048 packed tokens. Calibration can change on new data; evaluate thresholds for your application.

## Documentation and contributing

[Training](docs/TRAINING.md) · [Deployment](docs/DEPLOYMENT.md) · [API compatibility](docs/API.md) · [Data](docs/DATA.md) · [Contributing](CONTRIBUTING.md) · [Research roadmap](ROADMAP.md)

JevAny is independent of Jev and TypeSafe and includes no Jev weights or private implementation. It includes infrastructure adapted from [Kev](https://github.com/jaredpalmer/kev); see [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Code and starter data are Apache-2.0. Base models and upstream datasets retain their own terms.
