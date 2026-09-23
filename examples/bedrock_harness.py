#!/usr/bin/env python3
"""Compile a task with the existing Bedrock wrapper, then decide with JevAny."""
import argparse
import json

from jevany.bedrock import BedrockGenerator
from jevany.harness import HTTPDecisionClient, JevHarness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", help="optional AWS profile; standard AWS credentials are used when omitted")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--task", required=True)
    parser.add_argument("--evidence", required=True, help="JSON object, array, or scalar")
    parser.add_argument("--planner-sees-evidence", action="store_true",
                        help="send evidence values to the planner; this is not a prompt-injection security boundary")
    args = parser.parse_args()
    generator = BedrockGenerator(args.bedrock_model, profile=args.profile)
    harness = JevHarness(generator, HTTPDecisionClient(args.jev_url),
                         include_evidence_in_planner=args.planner_sees_evidence)
    result = harness.run(args.task, json.loads(args.evidence))
    result["planner"].update(model=args.bedrock_model, reasoning_enabled=False,
                             max_output_tokens=2048)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
