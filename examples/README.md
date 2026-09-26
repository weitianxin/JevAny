# Applications using one interface

Run from the repository root after [starting a server](../docs/DEPLOYMENT.md).
These examples adapt three scenarios from the existing
[showcase](../docs/CASES.md) into small, text-only applications.

```bash
python -m examples.inbox
python -m examples.sql_repair
python -m examples.service_recovery
```

| Example | What happens | Local effect |
|---|---|---|
| [Inbox](inbox.py) | Three messages are classified with choice and binary questions | Prints decisions; does not send or move mail |
| [SQL repair](sql_repair.py) | The model picks a provided query; SQLite runs it; an independent calculation checks totals | In-memory database |
| [Service recovery](service_recovery.py) | A bounded agent selects, prepares, canaries and promotes a replica | Local simulator, at most 12 decisions |

The SQL and recovery programs exit with status 1 if their checks fail. They
report the model's actual choices; they do not replace mistakes with a scripted
solution. New runs can differ from the selected successful showcase recordings.

Use your own server or load a checkpoint directly, with the same application code:

```bash
python -m examples.sql_repair --base-url http://127.0.0.1:8008
python -m examples.sql_repair --checkpoint runs/my-jev --device cuda
python -m examples.service_recovery \
  --checkpoint tianxinwei/JevAny-27B-SFT --device cuda --dtype bf16
```

`--checkpoint` needs the `local` extra; the released vision-capable checkpoint
also needs `multimodal`. The full base model must fit on the chosen device.

For your own environment, reuse `jevany.agent.run_episode`. It takes a callable
client, a goal, and an environment implementing `reset`, `step`, and
`get_all_actions`. The environment supplies finite actions and owns execution.
The [recovery example](service_recovery.py) shows this protocol.

[`request.json`](request.json) is a ready-to-send request:

```bash
jevany decide examples/request.json
jevany decide examples/request.json --checkpoint runs/my-jev
```

The earlier Bedrock [harness](bedrock_harness.py) and
[symbolic tree](bedrock_symbolic.py) examples remain available for applications
that use an LLM planner to construct bounded decisions. See
[INTEGRATIONS.md](../docs/INTEGRATIONS.md) for their additional dependencies.
