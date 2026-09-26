# Deploy a checkpoint

Install from the checkout using Python 3.12 or newer. The `local` extra loads
weights in Python; `serve` includes the same runtime plus FastAPI and Uvicorn.
Both use the adapter's recorded base model and temperature.
Training and serving resolve the same saved backbone adapter, tokenizer,
decision tokens and branch layout. Changing model families does not change
the request or response schema.

## Python

```bash
python -m pip install -e '.[local,multimodal]'
```

```python
from jevany import Choice, JevModel

model = JevModel.from_pretrained(
    "tianxinwei/JevAny-27B-SFT", device="cuda", dtype="bf16"
)
result = model.system_one(
    state="I was charged twice.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
        )
    },
)
print(result["answers"]["department"])
```

Use `JevModel.from_pretrained("runs/my-jev", model_name="my-jev")` for your
own checkpoint. Loading happens once; reuse the object between requests.
The common runtime serializes model/processor access and reuses eligible
state prefixes. Calling `model(request_dict)` accepts the complete HTTP body.

Configure inference without changing the checkpoint:

```python
from jevany import InferenceOptions, JevModel

model = JevModel.from_pretrained(
    "runs/my-jev",
    device="cuda",
    inference_options=InferenceOptions(
        max_state_tokens=2048,
        max_branch_tokens=4096,
        max_packed_tokens=8192,
        prefix_cache_size=4,
        prefix_min_tokens=384,
    ),
)
print(model.describe())
model.clear_cache()
```

`describe()` reports the resolved adapter, branch layout, context window, media
types, effective token limits and cache statistics. `clear_cache()` releases
prefixes and resets those statistics. Unknown architectures run without prefix
caching unless their adapter declares support; this optimization is optional.

## HTTP

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint tianxinwei/JevAny-27B-SFT \
  --device cuda --dtype bf16 --port 8008
```

To deploy your training output:

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

In another terminal:

```bash
jevany decide examples/request.json
curl http://127.0.0.1:8008/health
curl http://127.0.0.1:8008/v1/models
```

The model name in responses identifies the configured deployment, even when the
request uses an alias. Omit `model` or send `jevany-latest` to select the loaded
checkpoint; its exact reported ID is also accepted. Other IDs return HTTP 422.
This is a single-model server, and the request cannot load another checkpoint.
Missing checkpoints fail startup.

The server binds to loopback and has no authentication. Use `--host 0.0.0.0`
only inside a deployment with suitable network access controls; terminate
TLS and authentication at your gateway if exposing the API beyond a trusted host.

For an application embedding the server:

```python
from jevany.serve import create_app

app = create_app("runs/my-jev", device="cuda", model_name="my-jev")
```

The ASGI lifespan loads weights once at startup and releases the runtime on
shutdown. Alternatively, `create_app(model=model)` shares an existing `JevModel`,
including its lock and cache, with local callers. Each app has its own configured
runtime. `/health` returns 200 when ready and 503 when no runtime is loaded;
decision and model-discovery requests also return 503 while unavailable.
Use one Uvicorn worker per device: each worker loads a full model.

## Inference settings

Python and HTTP use `InferenceOptions`. Both `jevany serve` and local
`jevany decide --checkpoint` accept the corresponding CLI flags:

| Python field | CLI flag | Environment variable | Default |
|---|---|---|---|
| `max_state_tokens` | `--max-state-tokens` | `JEVANY_MAX_STATE_TOKENS` | 8192 |
| `max_branch_tokens` | `--max-branch-tokens` | `JEVANY_MAX_BRANCH_TOKENS` | 8192 |
| `max_packed_tokens` | `--max-packed-tokens` | `JEVANY_MAX_PACKED_TOKENS` | 8192 |
| `prefix_cache_size` | `--prefix-cache-size` | `JEVANY_PREFIX_CACHE` | 4 |
| `prefix_min_tokens` | `--prefix-min-tokens` | `JEVANY_PREFIX_MIN_TOKENS` | 384 |

A branch includes its shared state. State and branch limits are capped at the
backbone's context window; the packed limit covers the complete request, including
all branches. Oversized inputs are rejected without truncation. Set the cache size
to zero to disable reuse. Native media requests bypass the text prefix cache.
Prefix reuse is limited to validated FP32 backbones. BF16/FP16 models and the
recurrent Qwen3.5/3.8 architecture use complete forward passes. The runtime reports
the effective cache capability; requesting a cache size does not override it.

Explicit CLI flags override environment values. Passing an `InferenceOptions`
object in Python uses that whole object; otherwise settings are read from the
environment when loading. Invalid values fail before weights load. Weight-loading
controls remain in `jevany.checkpoint.LoadOptions`, shared with training and
evaluation.

## A lightweight HTTP client

The base package installs the schema, HTTP client and starter-data tools without
PyTorch. A caller connecting to another machine's server needs only:

```bash
python -m pip install -e .
```

```python
from jevany import JevClient, Noul

client = JevClient("http://127.0.0.1:8008")
result = client.system_one("The job has finished.", {"done": Noul(instructions="Is the job complete?")})
print(result["answers"]["done"]["noul"])
```

`JevClient` requires HTTPS for non-loopback endpoints. Pass `api_key` for your
gateway or the official hosted API. Invalid requests and responses raise
`ValueError`; transport and HTTP failures propagate from `urllib`.
Calls have a configurable 120-second default timeout and no automatic retries.

## Use the official SDK

The text request and answer envelopes follow the TypeSafe Jev API:

```bash
python -m pip install typesafe-sdk
```

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(
    api_key="local", base_url="http://127.0.0.1:8008", model="jevany-latest"
) as client:
    result = client.system_one(
        "I was charged twice.",
        {"department": Choice(criteria={"billing": "Payments", "shipping": "Delivery"})},
    )
    print(result.choices["department"].choice)
```

The official SDK has typed response objects. JevAny's Python clients return
ordinary dictionaries with the same wire fields. [API.md](API.md) describes
compatibility and the fields specific to JevAny.

## Checkpoints and hardware

| Checkpoint | Role |
|---|---|
| `tianxinwei/JevAny-27B-SFT` | Default released adapter |
| `tianxinwei/JevAny-27B-RLCR` | Experimental RLCR continuation |
| A training output directory | Your own adapter and pointer head |

An adapter is not a standalone copy of the base weights. First loading downloads
both the adapter and its separately distributed base, unless already cached.
Use `owner/repo@revision` to pin an adapter. For offline deployment, prepopulate
the Hugging Face cache and set `HF_HUB_OFFLINE=1`. `JEVANY_BASE_LOAD_PATH` can point
to a local base mirror while retaining the checkpoint's canonical provenance.

The runtime loads one full backbone on one device. The released 27B model needs
about 54 GB for BF16 base tensors alone, plus adapter and runtime memory.
CPU/MPS are available for backbones that fit, including smaller models you train.
Quantization and model sharding are
not implemented. BF16-trained checkpoints retain their recorded loading behavior.

## Native media and limits

For the server, install the `multimodal` extra and set `JEVANY_MEDIA_ROOT` to a
directory of inputs before startup. Paths in requests resolve inside that root;
network URLs, path escapes and oversized files are rejected. A media request
contains one question for the built-in adapters. In-process inference accepts
trusted local paths directly. Accepted media types come from the saved backbone
adapter; setting a media root does not enable vision in a text checkpoint.

```bash
JEVANY_MEDIA_ROOT="$PWD/media" jevany serve \
  --checkpoint tianxinwei/JevAny-27B-SFT --device cuda
```

See [DATA.md](DATA.md#native-media) for the request format and
`GET /v1/models` for the active limits and capabilities. The default packed limit is 8,192 tokens;
the validated training window is 2,048. Longer inputs are not an evaluated
capability. Confidence thresholds may need recalibration on your domain.

## Extend backbone support

Serving uses the same `backbone_adapter = "my_package.adapters:MyAdapter"` saved
by training. Install that trusted package in the serving environment; no HTTP
handler or client changes are needed. See [TRAINING.md](TRAINING.md#backbone-support)
for model-loading and media hooks.

An adapter's `inference_capabilities(config)` returns
`jevany.backbones.InferenceCapabilities`. It declares the context window,
prefix-cache support, accepted media types, and optional media question limit.
The built-in implementation reads the context window from the decoder config
and media types from the adapter. For example, a custom decoder can disable cache
reuse while retaining the other constraints:

```python
from dataclasses import replace
from jevany.backbones import BackboneAdapter

class MyAdapter(BackboneAdapter):
    def inference_capabilities(self, config):
        return replace(super().inference_capabilities(config), prefix_cache=False)
```

Only opt into caching after validating `new_cache()`, cache copying and batch
reordering, plus cropping for packed branches. Unknown text architectures use
independent rows and uncached inference by default. A model with a different
context layout should report its usable window explicitly.

The offline serving tests cover the seven maintained families, including recurrent
and MoE text models, Gemma 4 Unified and GLM native vision checkpoints, plus a
GPT-2 fixture for the generic adapter contract and a custom
adapter without cache support. They use tiny real architectures and native
processors to check Python/HTTP parity, media, cache behavior, limits and
lifecycle handling. They do not establish full-size model quality or GPU capacity.

```bash
python -m pytest tests/test_serving.py -q
```
