# Data

Training and inference share one JSON structure. Training rows add `label` and may add `target`. Store one object per line in UTF-8 JSONL.

## Question Types

### Choice

```json
{
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
    "billing": "Charges and payment problems",
    "shipping": "Delivery delays"
  },
  "label": "billing"
}
```

The label is one key from `criteria`.

### Noul

```json
{
  "type": "noul",
  "instructions": "Is manual review required?",
  "label": true
}
```

`noul` is a binary probability. Its label is `true` or `false`.

### Score

```json
{
  "type": "score",
  "instructions": "How urgent is this?",
  "criteria": ["low", "normal", "high"],
  "label": 2
}
```

The label is the zero-based level index.

## Soft Targets

Use a soft target when the evidence does not support one certain answer:

```json
{
  "type": "noul",
  "instructions": "Is the parcel late under the promised service level?",
  "label": false,
  "target": {"false": 0.5, "true": 0.5}
}
```

The `label` remains required for evaluation compatibility. Training uses `target` when present. Values are normalized after loading.

## Full Record

See [`examples/train.jsonl`](../examples/train.jsonl). `state` and `instructions` may be strings, objects, arrays, numbers, booleans, or null. Object field names are preserved as text labels. Every question should be answerable from the state and instructions alone.

When creating data:

- keep option keys stable and descriptions specific;
- include cases where information is genuinely missing;
- distinguish ambiguity from label noise;
- keep train and evaluation sources separate, then check normalized text hashes;
- preserve source, generator, prompt, verifier, and license metadata outside the model-facing fields;
- audit generated labels and counterfactual pairs before training.

## v0.1 Mixtures

The SFT mixture contains 12,576 source records:

| Group | Records |
|---|---:|
| 10 public classification sources | 10,000 |
| Compositional decisions | 1,680 |
| Policy decisions | 896 |

Online augmentation permuted choices and inserted none and distractor cases. Eligible rows could also emit a matched pair with the correct option present or absent.

The RLCR stage used exactly 8,192 records:

| Group | Records | Share |
|---|---:|---:|
| Hard public sources: Amazon, SST-5, Yelp, MNLI | 2,880 | 35.2% |
| Broad public replay | 2,040 | 24.9% |
| Compositional decisions | 1,024 | 12.5% |
| Policy decisions | 673 | 8.2% |
| Knowable/unknowable pairs | 1,575 | 19.2% |

The uncertainty block contains 525 unique rows repeated three times. The final mixture has zero normalized-text-hash overlap with the `transfer-v9` evaluation panel.

The repository publishes the mixture builder, not redistributed third-party datasets. Supply your licensed suite and uncertainty files to [`scripts/build_rlcr_mix.py`](../scripts/build_rlcr_mix.py).
