<p align="center">
  <img src="docs/hero.svg" alt="JevAny: reinforcement learning for adaptive decision systems" width="100%">
</p>

<p align="center">
  <a href="https://github.com/weitianxin/JevAny"><img alt="Release" src="https://img.shields.io/badge/release-v0.2-7c5cff"></a>
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97-models-ffb000"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="Tests" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

JevAny chooses among the options you give it. Send the current state, a question, and candidate answers; get back a selection and the probability of each option, without decoding answer text.

Use it to route a request, choose an agent's next action, or make a decision from image or video evidence. Your application supplies the available actions and executes the choice. For open-ended tasks, a [harness](#harness-and-symbolic-control) can first turn the task into explicit questions and options.

[Quick start](#quick-start) · [Results](#results) · [How it works](#how-it-works) · [Training](#train-your-own) · [Limits](#scope)

## Applications

Examples built with our [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) model.

[![JevAny choosing actions in robotics, laboratory, mobility, browser, and software tasks](docs/demos/jevany-cases.gif)](docs/demos/jevany-cases.gif)

[Explore the cases](docs/CASES.md) for each task, the constraints that decide the order of work, and the recorded decisions behind every trace.

## Quick start

You need Python 3.12 or newer. The commands below also need a CUDA GPU that holds the full 27B model in BF16 plus runtime memory, since the server loads the model onto one device. Leave local disk space for the base weights as well as the adapter.

The v0.2 release has two rank 16 LoRA adapters with a learned pointer head, both on a Qwen3.8-27B base:

| Checkpoint | When to use it |
|---|---|
| [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) | Start here. Stronger accuracy on both the development and transfer suites. |
| [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) | The reinforcement learning with calibration rewards (RLCR) experiment. No overall transfer gain over SFT so far. |

### Start the server

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[serve,multimodal]'

JEVANY_DTYPE=bf16 python -m jevany.serve \
  --run tianxinwei/JevAny-27B-SFT --device cuda --port 8008
```

The first start downloads the adapter and its separately distributed base weights from Hugging Face. `--run` also accepts a local checkpoint directory. Once the server is listening on port 8008, send a request from another terminal.

### Make a decision

This request asks two independent questions about the same ticket: which department should handle it, and how likely it is to need urgent review.

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

Read `answers.department.choice` for the selected department, `answers.department.probabilities` for the alternatives, and `answers.urgent.noul` for the probability of urgent review.

A text request can contain 1–64 questions about the same `state`. Three question types share the same endpoint:

| Type | You provide | You receive in `answers.<question_id>` |
|---|---|---|
| `choice` | Named options in a `criteria` object | The selected `choice`, all option `probabilities`, and normalized `confidence` |
| `noul` | A yes/no question | `noul`: the probability of **true**, from 0 to 1 |
| `score` | Ordered levels in a `criteria` list | The expected zero-based `score`, a `legend`, level `probabilities`, and `confidence` |

For `choice`, `confidence` rescales the highest option probability relative to a uniform distribution. Use the `probabilities` field when you need the probability of a particular option.

See the [request and training formats](docs/DATA.md) for complete examples. Image and video requests take a single question and need [media setup](#scope).

## One core, several systems

These integrations all call the same decision API. The status column says how far we took each one.

| System | What it does | Status |
|---|---|---|
| Jev-Judge | Typed choice, binary, and ordinal decisions with confidence | Released |
| Jev-Agent | Chooses actions in multi-step environments | Prototype evaluated |
| Jev-Harness | Lets an LLM compile open-ended tasks into bounded decisions | Prototype |
| Jev-Tool | Selects tools, execution modes, and escalation paths | Prototype |
| Jev-Symbolic | Runs LLM-authored, validated decision trees with JevAny at each node | Prototype |
| Jev-Test | Adapts from repeated samples without ground-truth labels | Negative result; see below |
| Jev-Image | Makes decisions from native image evidence | Evaluated with blank and shuffled controls |
| Jev-Video | Scores native video evidence | Evaluated on three temporal decision tasks |

Next priorities include harder agent tasks, long context, document images, broader temporal reasoning, computer use, coding, robotics, and guarded test-time updates. [ROADMAP.md](ROADMAP.md) lists the acceptance criteria.

## Results

SFT is the default checkpoint. RLCR improved development NLL by 0.005 and did not improve transfer accuracy.

The [v0.2 release record](results/release-v0.2.json) reports results on 1,004 development questions and 1,046 transfer questions. NLL (negative log-likelihood) penalizes low probability on the correct answer; lower is better.

| Model | Development accuracy | Development NLL ↓ | Transfer accuracy | MMLU-Pro | AI2D | MMMU |
|---|---:|---:|---:|---:|---:|---:|
| **JevAny-27B-SFT v2** | **90.34%** | 0.265 | **82.41%** | 73.0% | 86.0% | **68.0%** |
| JevAny-27B-RLCR v2 | 89.74% | **0.260** | 82.31% | **73.5%** | **87.0%** | 63.0% |
| Jev | n/a | n/a | 85.37% | 84.0% | n/a | n/a |

RLCR changed transfer accuracy by `-0.10` percentage points against SFT, with 3 fixes and 4 regressions. The paired 95% bootstrap interval is `[-0.58, 0.39]` points. The small development NLL gain did not survive independent calibration. Jev is a different hosted system evaluated through the same decision suite, not a weight-matched ablation.

Development accuracy and NLL exclude 100 VideoFeedback questions whose labels are all the same highest score. Those questions did exercise the video path, but a slice with one label cannot show temporal understanding, so we do not report it as a capability score. AI2D and MMMU use native images through the backbone's vision path. The training set also contains native A-OKVQA and ScienceQA images.

<p align="center">
  <img src="docs/results-v2.svg" alt="JevAny v2 evaluation overview" width="100%">
</p>

### Native image and video decisions

We evaluated the released SFT checkpoint with the real media, a neutral blank asset, and media shuffled between questions within each task. We shuffle by unique media group, so questions that share one image or video receive the same replacement. The media sensitivity gate requires full-media accuracy to exceed the stronger control by at least five points, with a positive paired media-group bootstrap interval. Passing the gate shows that the model reads the media; task accuracy is a separate question.

| Panel | Questions | Full media | Blank | Shuffled | Gain over strongest control |
|---|---:|---:|---:|---:|---:|
| MMStar clean image panel | 1,330 | **74.5%** | 29.0% | 28.9% | **+45.5** `[+42.4, +48.6]` |
| MVBench three-task video panel | 600 | **33.5%** | 12.3% | 15.3% | **+18.2** `[+13.9, +22.4]` |

| MVBench task | Questions | Full media | Blank | Shuffled | Gain over strongest control |
|---|---:|---:|---:|---:|---:|
| Fine-grained action | 200 | **45.0%** | 14.0% | 16.5% | **+28.5** `[+19.5, +37.5]` |
| Egocentric navigation | 200 | **43.5%** | 23.0% | 29.0% | **+14.5** `[+6.3, +22.4]` |
| Action antonym | 200 | **12.0%** | 0.0% | 0.5% | **+11.5** `[+7.0, +16.0]` |

From the image panel we dropped invalid choices and every item that matched the training, calibration, or development splits by media or by normalized question and unordered option text. No exact or perceptual media overlap, question-option overlap, or source-ID overlap remains. Its task-macro random and label-position baselines are 26.7% and 31.8%. The video panel passes the same zero-overlap checks. Its overall media gain is significant, though the action-antonym score is low and the video probabilities are badly calibrated.

Full metrics, per-task intervals, dataset revisions, checkpoint hashes, and control provenance are in [the image report](results/multimodal-image-v1.json) and [the video report](results/multimodal-video-v1.json). The internal evaluation checkpoint and public SFT release have identical LoRA and pointer-head tensors; [the equivalence record](results/release-equivalence-v0.2.json) accounts for the embedded release temperature. Upstream terms keep us from redistributing the benchmark media.

The examples below use synthetic media we made and released with this repository. Both cases per modality come from a predeclared set of three, and [the selection record](docs/demos/multimodal-demo.json) lists every probability. The two video cases put the same question to different clips, so only the motion separates the answers.

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/demos/jev-image-success.png" alt="JevAny reading a route diagram to pick the reachable destination" width="100%"><br><b>Jev-Image: which destination is still reachable</b></td>
    <td width="50%" align="center"><img src="docs/demos/jev-image-alternate.png" alt="JevAny reading bay status indicators to pick the bay needing inspection" width="100%"><br><b>Jev-Image: which bay needs inspection</b></td>
  </tr>
  <tr>
    <td width="50%" align="center"><img src="docs/demos/jev-video-success.gif" alt="JevAny tracking a cart that finishes at the west bay" width="100%"><br><b>Jev-Video: the cart finishes west</b></td>
    <td width="50%" align="center"><img src="docs/demos/jev-video-alternate.gif" alt="JevAny tracking a cart that finishes at the north bay" width="100%"><br><b>Jev-Video: the cart finishes north</b></td>
  </tr>
</table>

### Jev-Agent

<table>
  <tr>
    <td width="50%" align="center"><img src="docs/demos/jev-agent-frozen-lake.gif" alt="JevAny solving FrozenLake" width="100%"><br><b>FrozenLake</b><br>98% of 50 episodes solved</td>
    <td width="50%" align="center"><img src="docs/demos/jev-agent-sokoban.gif" alt="JevAny acting in Sokoban" width="100%"><br><b>Sokoban</b><br>48% of 50 episodes solved</td>
  </tr>
</table>

We expose only the legal actions at each step, and JevAny picks one without generating text. FrozenLake lets you recover from a bad step. Sokoban gives you no way to undo a push, so one choice can decide the episode.

### Jev-Test

We tested transductive adaptation without ground-truth labels. The parent produced 16 stochastic decisions per input, a strict majority became the pseudo label, and we dropped ties. We locked the protocol before scoring the adapted models against gold labels.

| Dataset | Parent accuracy | Pseudo-label SFT | Pseudo-label RLCR | Parent NLL | SFT NLL | RLCR NLL |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | **73.00%** | 72.00% | 72.00% | 0.942 | **0.930** | 0.933 |
| MuSR | 60.71% | **61.11%** | 60.98% | **1.122** | 1.552 | 1.483 |

MMLU-Pro NLL improved by 0.012 while accuracy fell a point. MuSR accuracy moved by at most 0.40 points while its NLL rose from 1.122 to 1.552. This self-training recipe is a negative result, and the code stays experimental. The [locked protocol](results/ttt-protocol-v1.json) and [release measurements](results/release-v0.2.json) document the full experiment.

## Harness and symbolic control

Jev-Harness uses an external LLM only as a task compiler. The planner sees an evidence schema by default, produces typed questions, and cannot replace the caller-owned state. JevAny then makes the bounded decision.

With the JevAny server running, install the Bedrock adapter:

```bash
python -m pip install -e '.[bedrock]'
```

Use standard AWS credentials with access to the chosen Bedrock model. This example uses the same planner model as the [recorded harness run](docs/demos/jev-harness.json):

```python
from jevany.bedrock import BedrockGenerator
from jevany.harness import HTTPDecisionClient, JevHarness

planner = BedrockGenerator("us.anthropic.claude-opus-4-7")
harness = JevHarness(planner, HTTPDecisionClient("http://127.0.0.1:8008"))
result = harness.run(
    "Choose an execution mode and decide whether rollback is required.",
    {"environment": "staging", "tests": "passed", "snapshot": "available"},
)
print(result["decision"]["answers"])
```

Jev-Symbolic asks an LLM to write a compact decision tree, validates branch coverage and acyclicity, then sends each internal node to JevAny. Every result records the outcome ID and complete branch trace.

For command-line entry points, see [the harness example](examples/bedrock_harness.py) and [the symbolic example](examples/bedrock_symbolic.py). The saved [harness](docs/demos/jev-harness.json) and [symbolic](docs/demos/jev-symbolic.json) outputs include the compiled requests and decisions.

## How it works

```text
state ───────────────┬─ question A ─ options ─ <decide> ─ probabilities A
                     ├─ question B ─ options ─ <decide> ─ probabilities B
                     └─ question C ─ options ─ <decide> ─ probabilities C
```

The Qwen3.8 backbone processes each question in its own causal row, repeating the shared state so that questions cannot influence one another. A learned pointer head compares the hidden state at `<decide>` with every option boundary. Softmax over those scores gives the answer distribution. A temperature fitted on a separate calibration partition adjusts the probabilities without changing the highest-scoring option.

Supervised fine-tuning (SFT) trains the adapter and pointer head with hard or soft targets while the base weights stay frozen. RLCR perturbs pointer logits, scores correctness and confidence, centers rewards within each proposal group, and anchors the update with supervised loss. It adapts the calibration reward from [Beyond Binary Rewards](https://arxiv.org/abs/2507.16806) to option probabilities. The policy operates on perturbed pointer logits rather than generated reasoning tokens. [docs/ALGORITHM.md](docs/ALGORITHM.md) gives the full objective and its differences from token-level GRPO.

## Train your own

Training uses the inference JSON shape plus a `label` on each question; optional `target` distributions provide soft labels. [examples/train.jsonl](examples/train.jsonl) is a small dataset for trying this format.

The example below uses eight CUDA GPUs, with a full model on each device. Set `--nproc_per_node` to the number of GPUs you intend to use; each must fit the model and its training memory.

```bash
python -m pip install -e '.[train]'

torchrun --nproc_per_node=8 -m jevany.train \
  --base Qwen/Qwen3.8-27B \
  --data examples/train.jsonl \
  --device cuda --lora 16 --weights_dtype bf16 --dtype bf16 \
  --checkpointing 1 \
  --out runs/my-sft
```

This trains an adapter and pointer head on the example data. Reproducing the published results also requires the original data and frozen evaluation suites, including `transfer-v9`, which we do not ship here.

The runtime supports multi-node distributed data parallel (DDP) training, distributed evaluation, regular checkpoints, and W&B. The [SFT](scripts/train_sft.sh) and [RLCR](scripts/train_rlcr.sh) templates record the release settings; [docs/DATA.md](docs/DATA.md) describes the mixtures and links to their builders.

## Scope

The released path accepts text, JSON-renderable state, native images, and native video. Multimodal requests currently support one isolated question and a bounded visual token budget.

The HTTP server disables media by default. To enable it, install the `multimodal` extra (included in the quick start) and set `JEVANY_MEDIA_ROOT` to a controlled local directory before starting the server. Requests may use only files inside that directory. The server rejects network media URLs and videos with no declared frame count, and it caps file size, total bytes, pixels, and declared video frames. See the [native media format](docs/DATA.md#native-media) for an example.

The validated text training window is 2,048 packed tokens. The server admits requests up to 8,192 packed tokens, but we have not validated longer contexts as a capability. Calibration can shift on new data; check confidence thresholds on your own evaluation set before using them to automate decisions.

## Attribution

JevAny includes Apache-2.0 infrastructure adapted from [Kev](https://github.com/jaredpalmer/kev). License notices are in [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). JevAny adds its own data, RL implementation, evaluation, agent, harness, and symbolic layers. It contains no Jev weights or private implementation.

## License

Apache-2.0. Base-model terms apply separately.
