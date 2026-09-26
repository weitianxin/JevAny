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
| **[训练](#训练)** | 训练数据、支持的基座、SFT/RLCR recipe 和多 GPU 训练入口 |
| **[推理与部署](#推理与部署)** | 预训练模型、本地 Python 推理和 HTTP API |

## 演示

以下案例使用 [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT)：

[![JevAny 在机器人、浏览器、软件、实验室和出行任务中选择动作](docs/demos/jevany-cases.gif)](docs/CASES.md)

[查看全部 30 个案例](docs/CASES.md)，了解任务和决策记录。这些是精选成功运行，不代表任务成功率。要测试自己的模型，可以运行[示例与测试环境](#示例与测试环境)。

## 安装

使用 Python 3.12 或更新版本。克隆仓库并创建环境：

```bash
git clone https://github.com/weitianxin/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
```

按用途选择依赖：

| 用途 | 安装命令 |
|---|---|
| 调用已有 HTTP 服务 | `python -m pip install -e .` |
| 训练文本模型 | `python -m pip install -e '.[train]'` |
| 在本地运行文本模型或启动 HTTP 服务 | `python -m pip install -e '.[serve]'` |
| 运行支持原生媒体输入的已发布 27B 模型 | `python -m pip install -e '.[serve,multimodal]'` |

只安装客户端不会引入 PyTorch。训练图片/视频模型时，使用 `.[train,multimodal]`。以下命令均在仓库根目录运行；具体模型的硬件要求见[预训练模型](#预训练模型)。

## 训练

### 训练数据

训练数据沿用推理时的 `state` 和 `questions`，并为每个问题增加 `label`。可选的软标签用于描述答案的概率分布。

| 数据 | 提供的内容 | 使用入口 |
|---|---|---|
| 随包入门数据 | 24 条合成训练记录和 8 条开发记录；文本输入，包含选择、二分类和评分问题 | `jevany data init --out data/starter` |
| 公开数据构建器 | 文本、图片和视频决策数据，来源包括 HelpSteer3、ScienceQA、A-OKVQA 和 VideoFeedback | `jevany data build-sft --help` · `jevany data build-rlcr --help` |
| 自己的数据 | 按统一 JSONL 格式添加标签的请求 | [格式与示例](docs/DATA.md) |

入门数据用于跑通流程。构建更大的数据集时，构建器会下载和转换上游数据，并记录来源版本、许可证和各分区的记录数。[数据构建指南](docs/TRAINING.md#data-beyond-the-starter) 提供了一个小规模纯文本构建示例。

训练前先准备并校验入门数据：

```bash
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
```

### SFT

监督微调让 Jev 模型学习带标签的决策。入门 recipe 使用 `Qwen/Qwen3.5-0.8B` 和 CUDA GPU，将 checkpoint 写入 `runs/my-jev`：

```bash
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

训练更新 LoRA adapter、决策头和新增决策 token 的 embedding，原有基座权重保持冻结。使用自己的数据时，添加 `--data data/my-domain.jsonl --out runs/domain-jev`。微调已发布的 Jev 模型可使用 [`recipes/finetune.toml`](recipes/finetune.toml)。

### RLCR

RLCR（Reinforcement Learning with Calibration Rewards）在 SFT 后继续训练，奖励同时考虑答案是否正确及其置信度。完成上面的 SFT recipe 后，运行：

```bash
jevany train --config recipes/rlcr.toml
```

该 recipe 从 `runs/my-jev` 加载模型，输出到 `runs/my-jev-rlcr`。修改 recipe 时，基座和 adapter 设置需与 SFT checkpoint 一致。RLCR 仍处于实验阶段，选择 checkpoint 前应在留出数据上比较准确率和校准效果。详见[训练目标](docs/ALGORITHM.md#rlcr)和[已发布模型的评测](#评测)。

### 支持的基座

同一训练器支持以下五个系列的官方基座：

| 系列 | 官方基座 |
|---|---|
| Qwen | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.5-0.8B` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Gemma | `google/gemma-4-31B-it` |
| Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512`, `mistralai/Ministral-3-14B-Instruct-2512-BF16` |
| Phi | `microsoft/Phi-4-reasoning-vision-15B` |

使用同一训练器切换基座：

```bash
jevany train --config recipes/sft.toml \
  --base meta-llama/Llama-3.1-8B-Instruct --out runs/llama-jev
```

Meta 权重需要已获授权的 Hugging Face 账号。原生视觉模型设置 `multimodal = true` 后可训练图片，并按模型能力支持视频。

可在本地 GPU 上训练，或用 `torchrun` 启动多卡训练；DDP 在每块 GPU 上保留完整基座。[训练指南](docs/TRAINING.md) 列出了各模型配置、多模态用法和自定义 adapter 接口。

## 预训练模型

| 模型 | 用途 |
|---|---|
| [JevAny-27B-SFT](https://huggingface.co/tianxinwei/JevAny-27B-SFT) | 默认发布模型 |
| [JevAny-27B-RLCR](https://huggingface.co/tianxinwei/JevAny-27B-RLCR) | RLCR 实验版本 |

两个版本都是基于 27B 视觉基座的 adapter，首次加载会另行下载基座权重。其 BF16 基座张量需要约 54 GB，此外还需 adapter 和运行时内存；展示案例使用 A100 80 GB GPU。自己训练的小模型使用同一 API，硬件需求由各自的基座决定。

运行时在单个设备上加载完整模型。[部署说明](docs/DEPLOYMENT.md#checkpoints-and-hardware) 包含硬件要求、离线加载和版本固定方法；[评测](#评测) 对比了两个已发布 checkpoint。

## 推理与部署

### Python API

[HTTP 服务](#http-服务)启动后，发送状态和带有候选选项的问题。返回值包含选中的选项及各选项的概率：

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

二分类问题使用 `Noul`，有序评分使用 `Score`。[API 文档](docs/API.md) 介绍了三种问题类型及兼容 Jev 的请求/返回格式。

### HTTP 服务

运行默认发布模型时，安装 `.[serve,multimodal]`，并使用满足 [27B 硬件要求](#预训练模型)的设备：

```bash
jevany serve --checkpoint tianxinwei/JevAny-27B-SFT \
  --device cuda --dtype bf16 --port 8008
```

部署前面 SFT 示例训练的小模型时，运行：

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

### 进程内推理

在应用中加载一次 checkpoint，复用上例中的 `state` 和 `questions`：

```python
from jevany import JevModel

jev = JevModel.from_pretrained("runs/my-jev", model_name="my-jev")
result = jev.system_one(state=state, questions=questions)
```

同一请求格式也适用于 `POST /v1/systemone`、`jevany decide examples/request.json` 和官方 TypeSafe SDK。各入口的用法见[部署指南](docs/DEPLOYMENT.md)。

原生图片/视频输入需要兼容的视觉 checkpoint，每个请求只支持一个问题。HTTP 媒体输入通过 `JEVANY_MEDIA_ROOT` 显式启用，详见[媒体配置与限制](docs/DEPLOYMENT.md#native-media-and-limits)。

## 示例与测试环境

服务启动后，可以用以下应用测试 checkpoint 在具体任务上的表现：

| 任务 | 环境 | 输出或成功判定 | 运行命令 |
|---|---|---|---|
| [收件箱分类](examples/inbox.py) | 三条本地示例消息 | 人工检查输出的文件夹和回复决策 | `python -m examples.inbox` |
| [SQL 修复](examples/sql_repair.py) | 内存 SQLite 数据库 | 查询汇总结果与独立计算一致 | `python -m examples.sql_repair` |
| [服务恢复](examples/service_recovery.py) | 本地副本模拟器 | 在 12 次决策内，让全部 100 个请求返回当前数据 | `python -m examples.service_recovery` |

SQL 修复和服务恢复检查失败时退出码为 1；收件箱分类打印决策供人工查看。任意示例加上 `--checkpoint runs/my-jev` 即可在进程内加载模型，也可以通过 `--base-url http://127.0.0.1:8008` 连接服务。

接入自己的环境时，实现 `reset`、`step` 和 `get_all_actions`，再通过 `jevany.agent.run_episode` 运行。[示例说明](examples/README.md) 介绍了接口；[Harness 与符号控制](docs/INTEGRATIONS.md) 提供可选的 LLM 规划层。

## 评测

已发布的 v0.2 模型在 1,046 道迁移问题上的结果：

| 模型 | 迁移集准确率 |
|---|---:|
| JevAny-27B-SFT | 82.41% |
| JevAny-27B-RLCR | 82.31% |

在这次评测中，RLCR 尚未带来整体迁移收益。[完整评测](docs/EVALUATION.md) 包含模型比较、图像/视频对照实验和测试时适配的负结果。

已验证的训练窗口为 2,048 个 packed tokens。新数据上的校准可能变化，部署前应在自己的留出任务上评估准确率和决策阈值。

## 文档与贡献

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API 兼容性](docs/API.md) · [数据](docs/DATA.md) · [评测](docs/EVALUATION.md) · [贡献指南](CONTRIBUTING.md) · [研究路线图](ROADMAP.md)

JevAny 独立于 Jev 和 TypeSafe，不包含 Jev 权重或私有实现。部分基础设施改编自 [Kev](https://github.com/jaredpalmer/kev)，归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。代码和入门数据采用 Apache-2.0；基础模型与上游数据集保留各自条款。
