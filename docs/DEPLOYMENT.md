# Deploy a checkpoint

Install from the checkout using Python 3.12 or newer. The `local` extra loads
weights in Python; `serve` includes the same runtime plus FastAPI and Uvicorn.
Both use the adapter's recorded base model and temperature.

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
request uses an alias. The request's `model` field does not load a different
checkpoint: this is a single-model server. Missing checkpoints fail startup;
there is no automatic fallback to another run.

The server binds to loopback and has no authentication. Use `--host 0.0.0.0`
only inside a deployment with suitable network access controls; terminate
TLS and authentication at your gateway if exposing the API beyond a trusted host.

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
about 54 GB for BF16 base tensors alone, plus adapter and runtime memory; prior
showcase inference used A100 80 GB GPUs. CPU/MPS are available for backbones that
fit, including smaller models you train. Quantization and model sharding are
not implemented. BF16-trained checkpoints retain their recorded loading behavior.

## Native media and limits

For the server, install the `multimodal` extra and set `JEVANY_MEDIA_ROOT` to a
directory of inputs before startup. Paths in requests resolve inside that root;
network URLs, path escapes and oversized files are rejected. A media request
contains one question. In-process inference accepts trusted local paths directly.

```bash
JEVANY_MEDIA_ROOT="$PWD/media" jevany serve \
  --checkpoint tianxinwei/JevAny-27B-SFT --device cuda
```

See [DATA.md](DATA.md#native-media) for the request format and
`GET /v1/models` for the active limits. The server admits 8,192 packed tokens;
the validated training window is 2,048. Longer inputs are not an evaluated
capability. Confidence thresholds may need recalibration on your domain.
