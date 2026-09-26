"""Adapt the inbox showcase into a text-only triage application."""
import json

from jevany import Choice, Noul
from ._common import client, parser

MESSAGES = [
    {"subject": "URGENT: account access", "sender_verified": False, "body": "Enter your password at this new link."},
    {"subject": "Design review", "sender_verified": True, "body": "Please review the draft today.", "thread_resolved": False},
    {"subject": "Launch issue", "sender_verified": True, "body": "Confirmed fixed. No further action needed.", "thread_resolved": True},
]


def run(decide) -> list[dict]:
    results = []
    for message in MESSAGES:
        results.append(decide.system_one(
            state={"message": message, "policy": "Quarantine unverified senders. Archive resolved threads. Focus unresolved actions due today."},
            questions={
                "folder": Choice(
                    instructions="Where should this message go?",
                    criteria={"quarantine": "Unverified sender", "archive": "Resolved thread", "focus": "Unresolved action due today"},
                ),
                "needs_reply": Noul(instructions="Does a verified sender need a reply about an unresolved action?"),
            },
        ))
    return results


def main():
    args = parser(__doc__).parse_args()
    print(json.dumps(run(client(args)), indent=2))


if __name__ == "__main__":
    main()
