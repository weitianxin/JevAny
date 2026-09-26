<p align="center">
  <img src="docs/title.png" alt="JevAny：从任意模型构建你的 Jev，部署到任意应用" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/collections/tianxinwei/jevany-adaptive-decision-systems-6ab2c941bcecb4d2c61d1326"><img alt="模型" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API 文档" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="示例" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/weitianxin/JevAny/actions"><img alt="测试" src="https://github.com/weitianxin/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

JevAny 用于训练和部署 Jev 风格的决策模型：你可以微调开源语言模型，也可以使用预训练 checkpoint。两者共用兼容 Jev 格式的 [Python 和 HTTP API](docs/API.md)，输入状态、问题与候选答案，即可获得选择及各选项的概率。

<p align="center">
  <img src="docs/hero.png" alt="JevAny 训练与部署流程：多模态数据、RLCR/SFT 训练、统一 API、测试环境与应用示例" width="100%">
</p>

| 从这里开始 | JevAny 提供什么 |
|---|---|
| **[训练模型](#训练模型)** | 入门数据、公开数据构建器、SFT / RLCR recipe、多 GPU 训练入口 |
| **[部署模型](#部署-checkpoint)** | 已发布 checkpoint、本地 Python 推理、HTTP 服务 |

## 看看它能做什么

以下案例使用 [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT)：

[![JevAny 在机器人、浏览器、软件、实验室和出行任务中选择动作](docs/demos/jevany-cases.gif)](docs/CASES.md)

[查看全部 30 个案例](docs/CASES.md)，了解任务和决策记录。这些是精选成功运行，不代表任务成功率。其中三个场景也提供了[使用统一接口的可运行代码](#运行示例)。

## 获取代码

需要 Python 3.12 或更新版本。随后按训练或部署需求安装对应依赖。

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

## 部署 checkpoint

已发布的 27B checkpoint 需要一块能容纳完整 BF16 基础模型和运行时开销的 GPU；展示案例使用 A100 80 GB。首次加载会下载 adapter 及单独分发的基础模型。你自己训练的小模型也使用相同接口。

```bash
python -m pip install -e '.[serve,multimodal]'

jevany serve --checkpoint tianxinwei/JevAny-27B-SFT \
  --device cuda --dtype bf16 --port 8008
```

服务启动后，从 Python 发起一次决策：

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

如果希望直接在应用进程中推理，用 `JevModel.from_pretrained(...)` 加载一次，再调用同样的 `system_one` 方法：

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
```

你也可以向 `POST /v1/systemone` 发送 JSON，运行 `jevany decide examples/request.json`，或让官方 TypeSafe SDK 连接这个服务。[部署指南](docs/DEPLOYMENT.md) 包含完整用法、硬件要求、离线加载和媒体输入说明。只连接已有服务时，`pip install -e .` 不会安装 PyTorch。

## 训练模型

Qwen、Llama、Gemma、Mistral 和 Phi 共用同一套训练接口。先用随包提供的数据和一个小型 Qwen 基座跑通流程：

```bash
python -m pip install -e '.[train]'

jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml

# 部署刚训练好的 checkpoint。
python -m pip install -e '.[serve]'
jevany serve --checkpoint runs/my-jev --model-name my-jev
```

入门数据包含 24 条原创合成训练记录和 8 条独立开发记录，每条都有选择、二分类和评分问题。这组数据用于熟悉流程；要训练有用的领域模型，请换成有代表性的业务数据。

SFT recipe 默认使用 `Qwen/Qwen2.5-0.5B` 和 CUDA GPU。修改 TOML 中的 `base`、`data`，或在命令行覆盖即可。训练记录在推理格式的每个问题上增加 `label`；训练器更新 LoRA adapter、pointer head 和新增决策 token 的 embedding，原有基础模型权重保持冻结。

| 下一步 | 命令或说明 |
|---|---|
| 使用自己的数据 | `jevany train --config recipes/sft.toml --data data/my-domain.jsonl --out runs/domain-jev` |
| 更换基座 | `jevany train --config recipes/sft.toml --base microsoft/Phi-4-mini-instruct --out runs/phi-jev` |
| 微调已发布模型 | [`recipes/finetune.toml`](recipes/finetune.toml) |
| 构建更大的数据集 | `jevany data build-sft --help` · [来源与格式](docs/DATA.md) |
| 多 GPU / 多机训练 | [`infra/train.sh`](infra/train.sh) |
| 实验校准奖励训练 | [`recipes/rlcr.toml`](recipes/rlcr.toml) |

[训练指南](docs/TRAINING.md) 介绍了基座 adapter、CPU 参数、Python 训练接口、评测和分布式启动。DDP 会在每块 GPU 上保留完整模型。

这五个系列的六个预训练基座均通过了 12 步 GPU 训练和 checkpoint 重载检查，最终 loss 低于初始值。详见[兼容性验证记录](results/backbone-smoke-v1.json)。

原生图片训练支持 Qwen VL、Llama Vision、Gemma 3、Pixtral 和转换后的 Phi-4 Multimodal；Qwen 还支持视频。选择视觉基座并设置 `multimodal = true`。参见[接入方式](docs/TRAINING.md#native-multimodal-training)和 [GPU 验证](results/multimodal-backbone-smoke-v1.json)。

## 运行示例

服务启动后：

```bash
python -m examples.inbox
python -m examples.sql_repair
python -m examples.service_recovery
```

| 示例 | 可以据此构建什么 |
|---|---|
| [收件箱分类](examples/inbox.py) | 分类消息并判断是否需要回复 |
| [SQL 修复](examples/sql_repair.py) | 选择查询，在 SQLite 中执行并核对结果 |
| [服务恢复](examples/service_recovery.py) | 在本地副本模拟器中运行多步智能体 |

给任意示例加上 `--checkpoint runs/my-jev`，即可切换为进程内推理。SQL 和服务恢复示例会如实报告检查失败。[示例说明](examples/README.md) 介绍各自的执行环境；[Harness 与符号控制](docs/INTEGRATIONS.md) 提供可选的 LLM 规划层。

## Checkpoint 与评测

| Checkpoint | 用途 | 迁移集准确率 |
|---|---|---:|
| [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) | 默认发布模型 | 82.41% |
| [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) | RLCR 实验版本 | 82.31% |

以上 v0.2 结果来自 1,046 道迁移问题，RLCR 尚未带来整体迁移收益。[完整评测](docs/EVALUATION.md) 保留模型比较、图像/视频对照实验和测试时适配的负结果；[算法说明](docs/ALGORITHM.md) 介绍训练目标。

原生图片和视频需要支持视觉的 checkpoint，每个请求只支持一个问题。HTTP 媒体输入通过 `JEVANY_MEDIA_ROOT` 显式启用。已验证的训练窗口为 2,048 个 packed tokens。新数据上的校准可能变化，自动决策阈值需要在自己的任务上评估。

## 文档与贡献

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API 兼容性](docs/API.md) · [数据](docs/DATA.md) · [贡献指南](CONTRIBUTING.md) · [研究路线图](ROADMAP.md)

JevAny 独立于 Jev 和 TypeSafe，不包含 Jev 权重或私有实现。部分基础设施改编自 [Kev](https://github.com/jaredpalmer/kev)，归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。代码和入门数据采用 Apache-2.0；基础模型与上游数据集保留各自条款。
