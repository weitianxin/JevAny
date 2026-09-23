#!/usr/bin/env python3
"""Generate an auditable decision tree with Bedrock and execute it with JevAny."""
import argparse
import json
import sys
from pathlib import Path

from jevany.harness import HTTPDecisionClient, json_object
from jevany.symbolic import JevTree, tree_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock-repo", default="/lustre-storage/fsx/tianxinwei/bedrock_model_query-main")
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", default="bedrock")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--task", required=True)
    parser.add_argument("--constraints", default="Use at most six binary or categorical decision nodes.")
    parser.add_argument("--evidence", required=True, help="JSON state evaluated by the generated tree")
    args = parser.parse_args()

    repository = Path(args.bedrock_repo).resolve()
    if not (repository / "orchestrator" / "llm_utils.py").exists():
        parser.error(f"Bedrock wrapper not found under {repository}")
    sys.path.insert(0, str(repository))
    from orchestrator.llm_utils import BedrockClient

    generator = BedrockClient({"name": args.bedrock_model, "profile": args.profile, "enable_reasoning": False})
    generated = generator.generate(tree_prompt(args.task, args.constraints),
                                   {"max_output_tokens": 4096, "temperature": 0.0})
    definition = json_object(generated["text"])
    result = JevTree.from_dict(definition).run(json.loads(args.evidence), HTTPDecisionClient(args.jev_url))
    print(json.dumps({"tree": definition, "result": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
