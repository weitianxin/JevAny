"""Validated symbolic decision trees with JevAny decisions at internal nodes."""
from dataclasses import dataclass
from typing import Callable

from .api import Choice, SystemOneRequest, validate_response


@dataclass(frozen=True)
class DecisionNode:
    question: Choice
    branches: dict[str, str]


@dataclass(frozen=True)
class DecisionTrace:
    node: str
    choice: str
    confidence: float
    next: str


class JevTree:
    def __init__(self, root: str, nodes: dict[str, DecisionNode], outcomes: dict[str, object]):
        self.root, self.nodes, self.outcomes = root, nodes, outcomes
        self._validate()

    @classmethod
    def from_dict(cls, value: dict):
        nodes = {}
        for name, node in value["nodes"].items():
            question = Choice.model_validate({"type": "choice", **node["question"]})
            nodes[name] = DecisionNode(question, dict(node["branches"]))
        return cls(value["root"], nodes, dict(value["outcomes"]))

    def _validate(self):
        if self.root not in self.nodes:
            raise ValueError("tree root must name a decision node")
        overlap = set(self.nodes) & set(self.outcomes)
        if overlap:
            raise ValueError(f"nodes and outcomes must have distinct names: {sorted(overlap)}")
        destinations = set(self.nodes) | set(self.outcomes)
        for name, node in self.nodes.items():
            if set(node.branches) != set(node.question.criteria):
                raise ValueError(f"node {name!r} must define one branch per choice")
            unknown = set(node.branches.values()) - destinations
            if unknown:
                raise ValueError(f"node {name!r} has unknown destinations: {sorted(unknown)}")
        visiting, visited = set(), set()

        def walk(name):
            if name in self.outcomes or name in visited:
                return
            if name in visiting:
                raise ValueError("decision tree contains a cycle")
            visiting.add(name)
            for target in self.nodes[name].branches.values():
                walk(target)
            visiting.remove(name)
            visited.add(name)

        walk(self.root)
        unreachable = set(self.nodes) - visited
        if unreachable:
            raise ValueError(f"decision tree has unreachable nodes: {sorted(unreachable)}")

    def run(self, state, decide: Callable[[dict], dict], *, model="jevany-27b", max_depth=None):
        node_name, trace = self.root, []
        max_depth = len(self.nodes) if max_depth is None else max_depth
        while node_name in self.nodes:
            if len(trace) >= max_depth:
                raise RuntimeError(f"decision tree exceeded max_depth={max_depth}")
            node = self.nodes[node_name]
            request = SystemOneRequest(state=state, model=model, questions={"branch": node.question}).model_dump(mode="json")
            answer = validate_response(request, decide(request))["answers"]["branch"]
            choice = answer["choice"]
            if choice not in node.branches:
                raise ValueError(f"decision backend returned unknown branch {choice!r}")
            target = node.branches[choice]
            trace.append(DecisionTrace(node_name, choice, float(answer["confidence"]), target))
            node_name = target
        return {"outcome": self.outcomes[node_name], "trace": [item.__dict__ for item in trace]}


def tree_prompt(task: str, constraints) -> str:
    return f"""Design a compact auditable decision tree for this task:
{task}

Constraints:
{constraints}

Return JSON only:
{{"root":"node_id","nodes":{{"node_id":{{"question":{{"instructions":"...","criteria":{{"choice":"description"}}}},"branches":{{"choice":"next_node_or_outcome"}}}}}},"outcomes":{{"outcome_id":{{}}}}}}
Every branch must lead to another node or a declared outcome. Do not decide the case itself."""
