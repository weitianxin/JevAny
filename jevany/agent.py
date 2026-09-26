"""Run JevAny as the policy for a discrete, multi-step environment."""
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

from .api import validate_response


class DiscreteEnvironment(Protocol):
    def reset(self, seed: int | None = None) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...
    def get_all_actions(self) -> list[Any]: ...


DecisionFunction = Callable[[dict], dict]


@dataclass(frozen=True)
class AgentStep:
    index: int
    observation: Any
    action: Any
    action_name: str
    confidence: float
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class Episode:
    seed: int | None
    success: bool
    reward: float
    steps: tuple[AgentStep, ...]

    def as_dict(self):
        return {**asdict(self), "steps": [asdict(step) for step in self.steps]}


def _observation(value):
    return value[0] if isinstance(value, tuple) else value


def action_request(goal, observation, actions: Mapping[str, str], history, model="jevany-latest"):
    return {
        "model": model,
        "state": {"goal": goal, "observation": observation, "recent_actions": history},
        "questions": {"action": {
            "type": "choice",
            "instructions": "Choose the next action that makes the most progress toward the goal.",
            "criteria": dict(actions),
        }},
    }


def run_episode(env: DiscreteEnvironment, decide: DecisionFunction, goal: str, *, seed=None,
                max_steps=32, history_limit=8, model="jevany-latest") -> Episode:
    """Run one episode against a RAGEN-style discrete environment.

    The environment owns transition and reward logic. JevAny only sees the rendered
    observation, recent actions, and the finite action set exposed at each turn.
    """
    observation = _observation(env.reset(seed=seed))
    trace, history, total_reward, success = [], [], 0.0, False
    action_names = getattr(env, "ACTION_LOOKUP", {})
    for index in range(max_steps):
        raw_actions = env.get_all_actions()
        if not raw_actions:
            break
        actions = {str(position): str(action_names.get(action, action))
                   for position, action in enumerate(raw_actions)}
        request = action_request(goal, observation, actions, history[-history_limit:], model)
        response = validate_response(request, decide(request))
        answer = response["answers"]["action"]
        action_key = str(answer["choice"])
        if action_key not in actions:
            raise ValueError(f"decision backend returned unknown action {action_key!r}")
        action = raw_actions[int(action_key)]
        next_observation, reward, done, info = env.step(action)
        step = AgentStep(index, observation, action, actions[action_key], float(answer["confidence"]),
                         float(reward), bool(done), dict(info))
        trace.append(step)
        total_reward += float(reward)
        success = success or bool(info.get("success"))
        history.append({"action": actions[action_key], "reward": float(reward), "effective": info.get("action_is_effective")})
        observation = _observation(next_observation)
        if done:
            break
    return Episode(seed, success, total_reward, tuple(trace))
