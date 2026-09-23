#!/usr/bin/env python3
"""Generate an auditable decision tree with Bedrock and execute it with JevAny."""
import argparse
import json

from jevany.bedrock import BedrockGenerator
from jevany.harness import HTTPDecisionClient, json_object
from jevany.symbolic import JevTree, tree_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", help="optional AWS profile; standard AWS credentials are used when omitted")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--task", required=True)
    parser.add_argument("--constraints", default="Use at most six binary or categorical decision nodes.")
    parser.add_argument("--evidence", required=True, help="JSON state evaluated by the generated tree")
    args = parser.parse_args()
    generator = BedrockGenerator(args.bedrock_model, profile=args.profile)
    generated = generator.generate(tree_prompt(args.task, args.constraints),
                                   {"max_output_tokens": 4096})
    definition = json_object(generated["text"])
    result = JevTree.from_dict(definition).run(json.loads(args.evidence), HTTPDecisionClient(args.jev_url))
    planner = {"model": args.bedrock_model, "reasoning_enabled": False, "max_output_tokens": 4096}
    planner.update({key: generated[key] for key in ("usage", "stop_reason") if generated.get(key) is not None})
    print(json.dumps({"planner": planner, "tree": definition, "result": result},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
