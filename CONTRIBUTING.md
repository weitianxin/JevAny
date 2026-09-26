# Contributing

Install the development dependencies with Python 3.12 or newer:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -m 'not server' -q
```

The framework tests create a tiny local Qwen backbone, run real SFT and RLCR
updates, reload the saved adapter, and check Python/HTTP/official-SDK contracts.
This is an integration check of the workflow, not a task-quality benchmark.
The unit tests also use a Qwen2.5 tokenizer, downloaded on first use.
Released-weight tests are optional and require `JEVANY_TEST_CHECKPOINT`.

For tests against an existing server:

```bash
JEVANY_BASE_URL=http://127.0.0.1:8008 python -m pytest tests/test_api.py -q
```

Keep JSON request/response compatibility when changing inference. Add labels only
to training records and keep them out of model-facing inputs. Data converters
should record source revisions and preserve evaluation separation.

The code is organized around user entry points:

| Location | Responsibility |
|---|---|
| `jevany/client.py`, `api.py` | Lightweight Python client and wire schema |
| `jevany/runtime.py`, `serve.py` | Shared local inference and HTTP deployment |
| `jevany/training.py`, `train.py` | Recipes, Python training entry point and training loop |
| `jevany/datasets/`, `data.py` | Bundled starter, public-source builders and JSONL validation |
| `recipes/`, `infra/` | Training configurations and scheduler-neutral launcher |
| `examples/` | Small applications using the public interfaces |
| `scripts/`, `results/` | Research/release tools and recorded measurements |
| `docs/` | Guides, evaluation records and showcase assets |

Build a distributable package with `python -m build`. Generated weights, runs,
downloaded data and build outputs are not source artifacts.
