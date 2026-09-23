<p align="center">
  <img src="docs/hero.svg" alt="JevAny: reinforcement learning for adaptive decision systems" width="100%">
</p>

<p align="center">
  <a href="https://github.com/weitianxin/JevAny"><img alt="Release" src="https://img.shields.io/badge/release-v0.2-7c5cff"></a>
  <a href="https://huggingface.co/collections/tianxinwei/jevany-6ab2c941bcecb4d2c61d1326"><img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97-models-ffb000"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="Tests" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

JevAny is a calibrated decision layer for reinforcement learning, agents, multimodal evidence, and model harnesses. It turns shared context into typed answers and option probabilities in one prefill pass. It does not generate an answer or a reasoning trace.

The current 27B release uses a Qwen3.8 backbone and contains two rank 16 LoRA checkpoints:

- [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) is the recommended general checkpoint.
- [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) is an experimental calibration-reward checkpoint.

Both require separately distributed base weights and the JevAny runtime.

## One Core, Several Systems

| System | What it does | Status |
|---|---|---|
| **Jev-Judge** | Typed choice, binary, and ordinal decisions with confidence | Released |
| **Jev-Agent** | Chooses actions in multi-step environments | Prototype evaluated |
| **Jev-Harness** | Lets an LLM compile open-ended tasks into bounded decisions | Prototype |
| **Jev-Tool** | Selects tools, execution modes, and escalation paths | Prototype |
| **Jev-Symbolic** | Runs LLM-authored, validated decision trees with JevAny at each node | Prototype |
| **Jev-Test** | Adapts from repeated samples without ground-truth labels | Research result |
| **Jev-Image** | Makes decisions from native image evidence | Trained and evaluated |
| **Jev-Video** | Scores native video evidence | Pipeline exercised; meaningful benchmark planned |

Next priorities are broader image and video evaluations, harder agent tasks, long context, computer use, coding, robotics, and guarded test-time updates. See [ROADMAP.md](ROADMAP.md) for acceptance criteria.

## Quick Start

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python -m venv .venv
source .venv/bin/activate
pip install -e '.[serve]'

hf download tianxinwei/JevAny-27B-SFT \
  --local-dir models/JevAny-27B-SFT

JEVANY_DTYPE=bf16 python -m jevany.serve \
  --run models/JevAny-27B-SFT --device cuda --port 8008
```

Send one state and up to 64 questions:

```bash
curl http://127.0.0.1:8008/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "model": "jevany-27b",
    "state": {
      "ticket": "The parcel is ten days late and the card was charged twice.",
      "customer_tier": "premium"
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this first?",
        "criteria": {
          "billing": "Charges and payment problems",
          "shipping": "Delivery delays and lost parcels",
          "returns": "Returns and exchanges"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this require human review within one hour?"
      }
    }
  }'
```

The response includes the selected answer, normalized confidence, and the full option distribution. [docs/DATA.md](docs/DATA.md) defines the request and training formats.

## Results

The v2 checkpoints were selected on separate development and transfer panels. Higher is better except NLL.

| Model | Development accuracy | Development NLL ↓ | Transfer accuracy | MMLU-Pro | AI2D | MMMU |
|---|---:|---:|---:|---:|---:|---:|
| **JevAny-27B-SFT v2** | **90.34%** | 0.265 | **82.41%** | 73.0% | 86.0% | **68.0%** |
| JevAny-27B-RLCR v2 | 89.74% | **0.260** | 82.31% | **73.5%** | **87.0%** | 63.0% |
| Jev | n/a | n/a | 85.37% | 84.0% | n/a | n/a |

RLCR changed transfer accuracy by `-0.10` percentage points against SFT, with 3 fixes and 4 regressions. The paired 95% bootstrap interval is `[-0.58, 0.39]` points. Its small development NLL gain did not transfer after independent calibration, so SFT remains the default. Jev is a different hosted system evaluated through the same decision suite, not a weight-matched ablation.

Development accuracy and NLL exclude 100 VideoFeedback questions whose labels are all the same highest score. The video path was exercised, but that slice is not evidence of temporal understanding and is not reported as a capability score. AI2D and MMMU use native images through the backbone's vision path. The training set also contains native A-OKVQA and ScienceQA images.

<p align="center">
  <img src="docs/results-v2.svg" alt="JevAny v2 evaluation overview" width="100%">
</p>

### Jev-Agent

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/demos/jev-agent-frozen-lake.gif" alt="JevAny solving FrozenLake" width="100%"><br><b>FrozenLake</b><br>98% success, 50 episodes</td>
    <td width="50%" align="center"><img src="docs/demos/jev-agent-sokoban.gif" alt="JevAny acting in Sokoban" width="100%"><br><b>Sokoban</b><br>48% success, 50 episodes</td>
  </tr>
</table>

Each step exposes only legal actions as options. JevAny selects an action without generating text. FrozenLake is nearly solved; Sokoban remains the useful hard case.

### Jev-Test

We tested transductive adaptation without ground-truth labels. For every input, the parent produced 16 stochastic decisions. A strict majority became the pseudo label; ties were rejected. The protocol was locked before post-adaptation gold scoring.

| Dataset | Parent accuracy | Pseudo-label SFT | Pseudo-label RLCR | Parent NLL | SFT NLL | RLCR NLL |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | **73.00%** | 72.00% | 72.00% | 0.942 | **0.930** | 0.933 |
| MuSR | 60.71% | **61.11%** | 60.98% | **1.122** | 1.552 | 1.483 |

MMLU-Pro calibration improved slightly while accuracy fell. MuSR accuracy moved by at most 0.40 points while calibration became much worse. This simple self-training recipe is therefore a negative result, not a release feature. Exact protocol and metrics are in [results/ttt-protocol-v1.json](results/ttt-protocol-v1.json) and [results/release-v0.2.json](results/release-v0.2.json).

## Harness And Symbolic Control

Jev-Harness uses an external LLM only as a task compiler. The planner sees an evidence schema by default, produces typed questions, and cannot replace the caller-owned state. JevAny then makes the bounded decision.

```python
from jevany.harness import HTTPDecisionClient, JevHarness

harness = JevHarness(bedrock_generator, HTTPDecisionClient("http://127.0.0.1:8008"))
result = harness.run(
    "Choose an execution mode and decide whether rollback is required.",
    {"environment": "staging", "tests": "passed", "snapshot": "available"},
)
```

Jev-Symbolic asks an LLM to author a compact decision tree, validates branch coverage and acyclicity, then sends each internal node to JevAny. Every result records the outcome ID and complete branch trace.

Install the optional Bedrock adapter and use standard AWS credentials:

```bash
pip install -e '.[bedrock]'
python examples/bedrock_harness.py --task 'Route this action' --evidence '{"risk":"low"}'
```

See [examples/bedrock_harness.py](examples/bedrock_harness.py), [examples/bedrock_symbolic.py](examples/bedrock_symbolic.py), and the saved [harness](docs/demos/jev-harness.json) and [symbolic](docs/demos/jev-symbolic.json) outputs.

## How It Works

```text
state ───────────────┬─ question A ─ options ─ <decide> ─ probabilities A
                     ├─ question B ─ options ─ <decide> ─ probabilities B
                     └─ question C ─ options ─ <decide> ─ probabilities C
```

The backbone reads the shared state and one causal row per question. A learned pointer head compares the hidden state at `<decide>` with every option boundary. Softmax over those scores gives the answer distribution.

SFT trains the adapter and pointer head with hard or soft targets. RLCR perturbs pointer logits, scores correctness and confidence, centers rewards within each proposal group, and anchors the update with supervised loss. It borrows the calibration reward from [Beyond Binary Rewards](https://arxiv.org/abs/2507.16806), but it is not token-level GRPO and does not generate reasoning or confidence tokens. [docs/ALGORITHM.md](docs/ALGORITHM.md) gives the full objective.

## Train Your Own

Training uses the inference JSON shape plus a `label` on each question:

```bash
torchrun --nproc_per_node=8 -m jevany.train \
  --base Qwen/Qwen3.8-27B \
  --data examples/train.jsonl \
  --lora 16 --weights_dtype bf16 --dtype bf16 \
  --out runs/my-sft
```

The runtime supports multi-node DDP, distributed evaluation, regular checkpoints, and W&B. The reproducible launch templates are [scripts/train_sft.sh](scripts/train_sft.sh) and [scripts/train_rlcr.sh](scripts/train_rlcr.sh).

## Scope

The released path accepts text, JSON-renderable state, native images, and native video. Multimodal requests currently support one isolated question and a bounded visual token budget. For safety, the HTTP server disables media by default. An operator can set `JEVANY_MEDIA_ROOT` to a controlled local directory; requests may then use only files inside that directory. Network media URLs are rejected, and file size, total bytes, pixels, and declared video frames are capped. Videos without a declared frame count are rejected. See [docs/DATA.md](docs/DATA.md) for an example.

The text training envelope is 2,048 packed tokens, and longer contexts have not been validated as a first-class capability. Confidence is an empirical measurement on the published distributions, not a deployment guarantee.

## Attribution

JevAny includes Apache-2.0 infrastructure adapted from [Kev](https://github.com/jaredpalmer/kev). License notices are in [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). JevAny adds its own data, RL implementation, evaluation, agent, harness, and symbolic layers. It contains no Jev weights or private implementation.

## License

Apache-2.0. Base-model terms apply separately.
