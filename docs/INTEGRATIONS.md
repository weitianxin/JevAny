# Harness and symbolic control

Jev-Harness uses an external LLM only as a task compiler. The planner sees an evidence schema by default, produces typed questions, and cannot replace the caller-owned state. JevAny then makes the bounded decision.

With the JevAny server running, install the Bedrock adapter:

```bash
python -m pip install -e '.[bedrock]'
```

Use standard AWS credentials with access to the chosen Bedrock model. This example uses the same planner model as the [recorded harness run](demos/jev-harness.json):

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

For command-line entry points, see [the harness example](../examples/bedrock_harness.py) and [the symbolic example](../examples/bedrock_symbolic.py). The saved [harness](demos/jev-harness.json) and [symbolic](demos/jev-symbolic.json) outputs include the compiled requests and decisions.
