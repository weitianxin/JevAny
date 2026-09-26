# Data

Training and inference share one JSON structure. Training rows add `label` and may add `target`. Store one object per line in UTF-8 JSONL.

For a dataset you can use immediately, run `jevany data init --out data/starter`.
It copies 24 original synthetic training records and 8 development records from
the installed package, with all three question types and provenance. This starter
is for learning the workflow. Validate it with
`jevany data validate data/starter/train.jsonl`.

For larger data, `jevany data build-sft` and `jevany data build-rlcr` expose the
public-source builders from an installed package. See [TRAINING.md](TRAINING.md)
for a text-only build and recipe commands.

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

The `label` remains required for evaluation compatibility. Training uses `target` when present. Values are normalized after loading. Targets may omit zero-weight options; unknown keys, negative or nonfinite weights, and zero total mass are rejected, along with labels outside the question's options.

## Native Media

Attach native image or video evidence at the request level. A multimodal request currently contains exactly one isolated question.

```json
{
  "state": {"study": "Inspect the diagram before answering."},
  "media": [{"type": "image", "uri": "cases/diagram.png"}],
  "questions": {
    "answer": {
      "type": "choice",
      "instructions": "Which component is connected to the battery?",
      "criteria": {"a": "Motor", "b": "Lamp"},
      "label": "b"
    }
  }
}
```

Training and frozen suites resolve relative paths against the JSONL directory. The HTTP server is stricter: media is disabled unless the operator sets `JEVANY_MEDIA_ROOT`, only local files contained by that root are accepted, and network URLs are rejected. File bytes, total request bytes, pixels, and declared video frames have configurable caps; videos without a declared frame count are rejected. Treat the media root as an upload quarantine, not a general filesystem directory.

## Full Record

See [`examples/train.jsonl`](../examples/train.jsonl). `state` and `instructions` may be strings, objects, arrays, numbers, booleans, or null. Object field names are preserved as text labels. Every question should be answerable from the state and instructions alone.

When creating data:

- keep option keys stable and descriptions specific;
- include cases where information is genuinely missing;
- distinguish ambiguity from label noise;
- keep train and evaluation sources separate, then check normalized text hashes;
- preserve source, generator, prompt, verifier, and license metadata outside the model-facing fields;
- audit generated labels and counterfactual pairs before training.

## v0.2 Mixtures

The SFT mixture contains 107,278 records and 127,012 questions:

| Group | Records |
|---|---:|
| HelpSteer3 preferences | 17,613 |
| Agent and tool decisions | 16,734 |
| A-OKVQA | 17,047 |
| VideoFeedback | 19,570 |
| QASC | 8,134 |
| CommonsenseQA | 9,741 |
| ScienceQA | 5,165 |
| Classification, policy, compositional, and ARC | 13,274 |

The 1,000-record calibration partition and 1,002-record development partition are separate. AI2D and MMMU appear only in those held-out partitions. Multimodal rows carry image or video URIs that are resolved and passed through the backbone's native processor. The training split contains 22,212 image records and 19,570 video records; each evaluation partition contains 500 media records.

The selected RLCR mixture contains exactly 40,000 records and 46,044 questions:

| Group | Records |
|---|---:|
| Hard reasoning | 6,000 |
| Many-choice reasoning | 6,000 |
| Agent and tool decisions | 5,000 |
| HelpSteer3 preferences | 5,000 |
| Mathematical reasoning | 4,000 |
| Medical reasoning | 4,000 |
| Image-derived decisions | 4,000 |
| Video-derived decisions | 2,000 |
| Core replay | 4,000 |

The mixture includes 5,000 HelpSteer3 rows, 6,000 eight-option QASC rows, 4,000 AQuA-RAT rows, 4,000 MedMCQA rows, and broad replay to limit drift. It has zero normalized-text-hash overlap with the `transfer-v9` evaluation panel.

The selected VideoFeedback `real` configuration is single-class in this conversion: every one of its five score dimensions maps to level 3. Those rows exercise the native video data and model path but do not form a meaningful accuracy benchmark. Release headline development metrics exclude the 100-question VideoFeedback slice. A label-balanced temporal benchmark is required before claiming video understanding.

The repository publishes mixture builders, not redistributed third-party datasets. Review each upstream license before downloading, training, or redistributing converted records. Build SFT data with [`scripts/build_v2_data.py`](../scripts/build_v2_data.py) and RL data with [`scripts/build_v2_rlcr_mix.py`](../scripts/build_v2_rlcr_mix.py).

## Test-Time Data

Jev-Test removes every `label` and `target` before inference. It samples the parent decision 16 times at temperature 0.8, accepts only a strict majority, and rejects ties. The adapted checkpoint never sees ground truth. Gold labels remain in the immutable source suite and are opened once after both SFT and RL adaptations finish.

This protocol accepted 153 of 200 MMLU-Pro inputs and 732 of 756 MuSR inputs. The exact hashes and settings are recorded in [`results/ttt-protocol-v1.json`](../results/ttt-protocol-v1.json).
