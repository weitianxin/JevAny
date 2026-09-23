#!/usr/bin/env python3
"""Compile a task with the existing Bedrock wrapper, then decide with JevAny."""
import argparse
import json
import sys
from pathlib import Path

from jevany.harness import HTTPDecisionClient, JevHarness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock-repo", default="/lustre-storage/fsx/tianxinwei/bedrock_model_query-main")
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", default="bedrock")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--task", required=True)
    parser.add_argument("--evidence", required=True, help="JSON object, array, or scalar")
    parser.add_argument("--planner-sees-evidence", action="store_true",
                        help="send evidence values to the planner; this is not a prompt-injection security boundary")
    args = parser.parse_args()

    repository = Path(args.bedrock_repo).resolve()
    if not (repository / "orchestrator" / "llm_utils.py").exists():
        parser.error(f"Bedrock wrapper not found under {repository}")
    sys.path.insert(0, str(repository))
    from orchestrator.llm_utils import BedrockClient

    generator = BedrockClient({
        "name": args.bedrock_model,
        "profile": args.profile,
        "enable_reasoning": False,
    })
    harness = JevHarness(generator, HTTPDecisionClient(args.jev_url),
                         include_evidence_in_planner=args.planner_sees_evidence)
    print(json.dumps(harness.run(args.task, json.loads(args.evidence)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
