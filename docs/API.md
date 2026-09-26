# One decision API

`JevClient`, `JevModel`, and `POST /v1/systemone` share a request/answer envelope.
Training data adds a label and optionally a soft target to each question.
JevAny is an independent implementation of Jev-style decision models.

## Request and response

```json
{
  "model": "jevany-latest",
  "state": "I was charged twice.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payment problems", "shipping": "Delivery problems"}
    },
    "urgent": {"type": "noul", "instructions": "Does this need urgent review?"},
    "priority": {"type": "score", "instructions": "Assign priority.", "criteria": ["low", "normal", "high"]}
  }
}
```

| Question | Required criteria | Answer fields |
|---|---|---|
| `choice` | Map of option names to descriptions | `type`, `choice`, `probabilities`, `confidence` |
| `noul` | Optional `true` / `false` descriptions | `type`, `noul` (probability of true) |
| `score` | Ordered level descriptions | `type`, `score`, `legend`, `probabilities`, `confidence` |

Responses contain `model`, `answers` keyed by the same question IDs, and `usage`
with `input_tokens` and `output_tokens`. JevAny also reports `latency_ms`.
`output_tokens` counts the serialized answer; the model does not decode text.
Score is the expected zero-based level and can lie between integer levels.
Score legends preserve structured criteria as JSON.

```python
from jevany import Choice, JevClient, Noul, Score

client = JevClient()
result = client.system_one(
    {"ticket": "I was charged twice."},
    {
        "department": Choice(criteria={"billing": "Payments", "shipping": "Delivery"}),
        "urgent": Noul(instructions="Does this need urgent review?"),
        "priority": Score(instructions="Assign priority.", criteria=["low", "normal", "high"]),
    },
)
print(result["answers"]["department"]["choice"])
print(result["answers"]["department"]["probabilities"])
```

Raw question dictionaries require the `type` discriminator. The Python question
constructors fill it in. Instructions may be omitted; explicit instructions
usually make the intended decision clearer. A request contains 1–64 questions.
Each question is isolated from siblings during inference.

The default model selector is `jevany-latest`, an alias for the one loaded
checkpoint. Its reported model ID is also accepted. Unknown selectors raise
`ValueError` locally and return HTTP 422. Responses always identify the loaded
model. Callers serving a custom checkpoint should omit `model`, use the alias,
or send the ID reported by `GET /v1/models`.

`JevModel.describe()` and `GET /v1/models` expose the resolved backbone adapter,
branch layout, context window, media types, active token limits and prefix-cache
support/statistics. The HTTP description also includes the server's media-file
policy. Requests exceeding those limits return HTTP 422 without truncation.
See [DEPLOYMENT.md](DEPLOYMENT.md#inference-settings) for configuration.

## Compatibility with Jev

The reference is TypeSafe's [HTTP API](https://docs.typesafe.ai/api) and
[Python SDK](https://docs.typesafe.ai/sdk/python/usage), checked on September 26,
2026. Contract tests exercise the official SDK against JevAny's local server.

Compatibility covers the text JSON envelope, three question types, option keys,
probability fields, and zero-based scores. It does not imply the same model
weights, predictions, calibration, or hosted service behavior.

| Detail | JevAny behavior |
|---|---|
| Model IDs | Deployment-specific identity; one checkpoint per server |
| Authentication | None on the local server; optional at a gateway |
| State | Text or structured JSON; additionally accepts top-level scalar/null values |
| Choice | 1–255 options |
| Score | 2–255 levels; stay within 2–10 for the documented hosted API contract |
| Media | JevAny-specific `media: [{type, uri}]` local-file extension |
| Confidence | Computed locally from the option distribution; exact hosted formulas are not guaranteed |
| Usage | Local token accounting, not hosted billing parity |

Choice confidence is `(max_probability - 1/K) / (1 - 1/K)`, or 1 for a single
option. Score confidence is `1 - E[abs(level - mode)] / (L - 1)`. Use raw
probabilities when applying your own thresholds. [DEPLOYMENT.md](DEPLOYMENT.md)
shows both JevAny's client and the official SDK.

## Training labels

Add `"label": "billing"` to a choice question, a boolean label to a noul question,
or an integer level index to a score question. Optional `"target"` maps option
keys to nonnegative weights; missing keys have zero weight and the vector is
normalized. Unknown keys, nonfinite weights, zero total mass, and invalid labels
are rejected. Labels and targets are stripped before model encoding.

Save one request per line and run `jevany data validate your-data.jsonl`.
See [DATA.md](DATA.md) for full examples and media path resolution.
