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
| **[Training](#training)** | Training data, supported backbones, SFT/RLCR recipes, and a multi-GPU launcher |
| **[Inference & Serving](#inference--serving)** | Pretrained models, local Python inference, and an HTTP API |

## Demos

Examples built with [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT):

[![JevAny choosing actions across robotics, browser, software, laboratory and mobility tasks](docs/demos/jevany-cases.gif)](docs/CASES.md)

[Explore all 30 cases](docs/CASES.md), with task descriptions and decision records. These are selected successful runs, not a success-rate benchmark. To try your own model, run the [examples and test environments](#examples--test-environments).

## Installation

Use Python 3.12 or newer. Clone the repository and create an environment:

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

Choose the dependencies for your use case:

| Use case | Install |
|---|---|
| Call an existing HTTP server | `python -m pip install -e .` |
| Train a text model | `python -m pip install -e '.[train]'` |
| Run a text model locally or serve it over HTTP | `python -m pip install -e '.[serve]'` |
| Run the released 27B models with native media support | `python -m pip install -e '.[serve,multimodal]'` |

The client-only installation does not install PyTorch. For image/video training, use `.[train,multimodal]`. Run the commands below from the repository root; model-specific hardware requirements are listed under [Pretrained Models](#pretrained-models).

## Training

### Training Data

Training uses the same `state` and `questions` as inference, with a `label` added to each question. Optional soft targets describe a distribution over answers.

| Data | What is available | Start here |
|---|---|---|
| Included starter | 24 synthetic training records and 8 development records; text inputs with choice, binary and score questions | `jevany data init --out data/starter` |
| Public-source builders | Text, image and video decisions, including HelpSteer3, ScienceQA, A-OKVQA and VideoFeedback | `jevany data build-sft --help` · `jevany data build-rlcr --help` |
| Your own data | Labelled requests in the shared JSONL format | [Format and examples](docs/DATA.md) |

The starter is for learning the workflow. For larger datasets, the builders download and convert upstream data and record source versions, licenses and split counts. See the [data-building guide](docs/TRAINING.md#data-beyond-the-starter) for a small text-only build.

Prepare and validate the starter before training:

```bash
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
```

### SFT

Supervised fine-tuning fits a Jev model to labelled decisions. The starter recipe uses `Qwen/Qwen3.5-0.8B` on a CUDA GPU and writes the checkpoint to `runs/my-jev`:

```bash
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

Training updates LoRA adapters, the decision head and any added decision-token embeddings; the original base weights stay frozen. To use your own data, add `--data data/my-domain.jsonl --out runs/domain-jev`. To adapt a released Jev model, use [`recipes/finetune.toml`](recipes/finetune.toml).

### RLCR

Reinforcement Learning with Calibration Rewards continues SFT with a reward based on both correctness and confidence. After completing the SFT recipe above, run:

```bash
jevany train --config recipes/rlcr.toml
```

This recipe loads `runs/my-jev` and writes `runs/my-jev-rlcr`. Keep the base and adapter settings consistent with the SFT checkpoint when changing the recipe. RLCR is experimental; compare accuracy and calibration on held-out data before choosing a checkpoint. See the [training objective](docs/ALGORITHM.md#rlcr) and [released-model evaluation](#evaluation).

### Supported Backbones

These official backbones share the same trainer:

| Family | Official bases |
|---|---|
| Qwen | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.5-0.8B` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Gemma | `google/gemma-4-31B-it` |
| Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512`, `mistralai/Ministral-3-14B-Instruct-2512-BF16` |
| Phi | `microsoft/Phi-4-reasoning-vision-15B` |

Select a different base with the same trainer:

```bash
jevany train --config recipes/sft.toml \
  --base meta-llama/Llama-3.1-8B-Instruct --out runs/llama-jev
```

Meta weights require approved Hugging Face access. Set `multimodal = true` for native vision training; image and video support follows the selected base.

Train on local GPUs or with `torchrun`; DDP keeps a full base on each GPU. The [training guide](docs/TRAINING.md) covers model settings, native media and custom adapters.

## Pretrained Models

| Model | Intended use |
|---|---|
| [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) | Default released model |
| [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) | Experimental RLCR continuation |

Both releases are adapters over a 27B vision-capable base, downloaded separately on first load. Their BF16 base tensors require about 54 GB, plus adapter and runtime memory; the showcase used A100 80 GB GPUs. Smaller models you train use the same API with their own hardware requirements.

The runtime loads one full model on one device. [Deployment details](docs/DEPLOYMENT.md#checkpoints-and-hardware) cover hardware, offline loading and revision pinning; [Evaluation](#evaluation) compares the released checkpoints.

## Inference & Serving

### Python API

With an [HTTP server](#http-server) running, send a state and a question with named options. The answer contains the selected option and each option's probability:

```python
from jevany import Choice, JevClient

state = {"ticket": "I was charged twice. Please help."}
questions = {
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
    ),
}

jev = JevClient("http://127.0.0.1:8008")
result = jev.system_one(state=state, questions=questions)
answer = result["answers"]["department"]
print(answer["choice"])
print(answer["probabilities"])
```

Use `Noul` for binary questions and `Score` for ordered levels. The [API reference](docs/API.md) describes all three question types and the Jev-compatible request/answer format.

### HTTP Server

For the default released model, install `.[serve,multimodal]` and use hardware that meets the [27B requirements](#pretrained-models):

```bash
jevany serve --checkpoint tianxinwei/JevAny-27B-SFT \
  --device cuda --dtype bf16 --port 8008
```

To serve the smaller model from the SFT example instead:

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

### In-Process Inference

Load a checkpoint once in your application and reuse `state` and `questions` from the example above:

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
result = jev.system_one(state=state, questions=questions)
```

The same request format also works with `POST /v1/systemone`, `jevany decide examples/request.json`, and the official TypeSafe SDK. See the [deployment guide](docs/DEPLOYMENT.md) for each option.

Native image/video inputs require a compatible vision checkpoint and one question per request. Enable HTTP media inputs explicitly through `JEVANY_MEDIA_ROOT`; see [media setup and limits](docs/DEPLOYMENT.md#native-media-and-limits).

## Examples & Test Environments

With a server running, use these applications to try a checkpoint on a concrete task:

| Task | Environment | Result or success criterion | Run |
|---|---|---|---|
| [Inbox triage](examples/inbox.py) | Three local sample messages | Inspect the printed folder and reply decisions | `python -m examples.inbox` |
| [SQL repair](examples/sql_repair.py) | In-memory SQLite database | Query totals match an independent calculation | `python -m examples.sql_repair` |
| [Service recovery](examples/service_recovery.py) | Local replica simulator | Serve all 100 requests with current data within 12 decisions | `python -m examples.service_recovery` |

SQL repair and service recovery exit with status 1 when their checks fail. Inbox triage prints decisions for manual inspection. Pass `--checkpoint runs/my-jev` to any example to load your model in-process, or `--base-url http://127.0.0.1:8008` to use a server.

To add an environment, implement `reset`, `step` and `get_all_actions` and run it with `jevany.agent.run_episode`. [Example instructions](examples/README.md) explain the interface; [harness and symbolic integrations](docs/INTEGRATIONS.md) add optional LLM planning.

## Evaluation

The released v0.2 models were evaluated on 1,046 transfer questions:

| Model | Transfer accuracy |
|---|---:|
| JevAny-27B-SFT | 82.41% |
| JevAny-27B-RLCR | 82.31% |

RLCR has not shown an overall transfer gain in this evaluation. [Evaluation details](docs/EVALUATION.md) include the full comparison, image/video controls and negative test-time adaptation results.

The validated training window is 2,048 packed tokens. Calibration can change on new data; evaluate accuracy and decision thresholds on your own held-out tasks before deployment.

## Documentation and Contributing

[Training](docs/TRAINING.md) · [Deployment](docs/DEPLOYMENT.md) · [API compatibility](docs/API.md) · [Data](docs/DATA.md) · [Evaluation](docs/EVALUATION.md) · [Contributing](CONTRIBUTING.md) · [Research roadmap](ROADMAP.md)

JevAny is independent of Jev and TypeSafe and includes no Jev weights or private implementation. It includes infrastructure adapted from [Kev](https://github.com/jaredpalmer/kev); see [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Code and starter data are Apache-2.0. Base models and upstream datasets retain their own terms.
