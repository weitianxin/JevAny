#!/usr/bin/env python3
"""Evaluate JevAny on RAGEN discrete-action environments."""
import argparse
import json
import statistics
import sys
from pathlib import Path

from jevany.agent import run_episode
from jevany.harness import HTTPDecisionClient
from jevany.suite import write_json


GOALS = {
    "frozen_lake": "Reach G without entering O. P is the player, _ is safe ice, O is a hole, and G is the goal.",
    "sokoban": "Push every X box onto an O target. # is a wall, P is the player, and √ is a box on target.",
}


def environment(name, repository):
    sys.path.insert(0, str(repository))
    if name == "frozen_lake":
        from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
        from ragen.env.frozen_lake.env import FrozenLakeEnv
        return FrozenLakeEnv(FrozenLakeEnvConfig(size=4, p=0.9, is_slippery=False,
                                                  success_rate=1.0, observation_format="grid_coord"))
    if name == "sokoban":
        from ragen.env.sokoban.config import SokobanEnvConfig
        from ragen.env.sokoban.env import SokobanEnv
        return SokobanEnv(SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=64,
                                            search_depth=30, observation_format="grid_coord"))
    raise ValueError(f"unsupported RAGEN environment: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragen-repo", default="../RAGEN")
    parser.add_argument("--environment", choices=sorted(GOALS), required=True)
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    repository = Path(args.ragen_repo).resolve()
    if not (repository / "ragen" / "env" / "base.py").exists():
        parser.error(f"RAGEN checkout not found under {repository}")

    env = environment(args.environment, repository)
    decide = HTTPDecisionClient(args.jev_url)
    episodes = [run_episode(env, decide, GOALS[args.environment], seed=args.seed + index,
                            max_steps=args.max_steps).as_dict()
                for index in range(args.episodes)]
    output = {
        "environment": args.environment,
        "ragen_repo": str(repository),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "mean_reward": statistics.fmean(item["reward"] for item in episodes),
        "mean_steps": statistics.fmean(len(item["steps"]) for item in episodes),
        "traces": episodes,
    }
    write_json(args.out, output)
    print(json.dumps({key: value for key, value in output.items() if key != "traces"}, indent=2))


if __name__ == "__main__":
    main()
