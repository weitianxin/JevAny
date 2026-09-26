"""Run a bounded recovery loop against a local replica simulator."""
import copy
import json

from jevany.agent import run_episode
from ._common import client, parser


class Recovery:
    ACTION_LOOKUP = {
        "select_a": "Select replica A",
        "select_b": "Select replica B",
        "warm_a": "Increase replica A capacity from 60 to 120 requests",
        "sync_b": "Catch replica B up to the current version",
        "canary": "Check 10 requests on the selected replica",
        "promote": "Move the 100-request workload to the selected replica, requiring a current passing canary",
        "probe": "Check the promoted route against all 100 requests and finish",
    }

    def reset(self, seed=None):
        self.state = {
            "replicas": {"a": {"capacity": 60, "lag": 0}, "b": {"capacity": 120, "lag": 4}},
            "selected": None, "active": "primary", "canary": None,
            "workload": 100, "feedback": "The primary is failing. Restore all requests with current data.",
        }
        self.done = False
        return copy.deepcopy(self.state)

    def get_all_actions(self):
        return [] if self.done else list(self.ACTION_LOOKUP)

    def step(self, action):
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable action: {action}")
        state, success = self.state, False
        state["feedback"] = "Action completed."
        if action.startswith("select_"):
            state["selected"], state["canary"] = action[-1], None
        elif action in ("warm_a", "sync_b"):
            if action == "warm_a":
                state["replicas"]["a"]["capacity"] = 120
            else:
                state["replicas"]["b"]["lag"] = 0
            state["canary"] = None
        elif action == "canary":
            target = state["selected"]
            passing = target is not None and state["replicas"][target]["lag"] == 0
            state["canary"] = {"target": target, "passed": passing}
        elif action == "promote":
            canary = state["canary"]
            if canary and canary["passed"] and canary["target"] == state["selected"]:
                state["active"] = state["selected"]
            else:
                state["feedback"] = "Promotion rejected: run a passing canary after the last change."
        elif action == "probe":
            active = state["replicas"].get(state["active"])
            success = bool(active and active["lag"] == 0 and active["capacity"] >= state["workload"])
            self.done = True
            state["feedback"] = "All 100 requests returned current data." if success else "Recovery failed full-workload checks."
        return copy.deepcopy(state), float(success), self.done, {"success": success}


def run(decide):
    return run_episode(
        Recovery(), decide,
        "Recover the service: serve all 100 requests with no replication lag. Check the full workload after promotion.",
        max_steps=12,
    ).as_dict()


def main():
    args = parser(__doc__).parse_args()
    result = run(client(args))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
