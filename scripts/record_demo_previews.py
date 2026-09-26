"""Record real checkpoint decisions in the native game environments.

Start a JevAny server, then run with the demo extra installed:
    python scripts/record_demo_previews.py --base-url http://127.0.0.1:8008 \
        --model tianxinwei/JevAny-27B-SFT --media-root /tmp/jevany-media
The robot replay is maintained separately.
"""
import argparse
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jevany.client import DecisionClient, JevClient
from jevany.demos.server import DemoApplication


def record(client: DecisionClient, case: str, seed: int, root: Path, *,
           media_root: Path | None = None, images: bool = True) -> dict:
    """Record one episode, executing each validated model choice exactly once."""
    if case not in ("crafter", "doom"):
        raise ValueError(f"unknown game {case!r}; choose crafter or doom")
    root.mkdir(parents=True, exist_ok=True)
    app = DemoApplication(client, images=images, media_root=media_root)
    staging = TemporaryDirectory(prefix=f".{case}-", dir=root)
    steps, frame_index = [], 0
    try:
        directory = Path(staging.name) / "recording"
        directory.mkdir()
        snapshot = app.start(case, seed)
        while True:
            frames = []
            for uri in snapshot["frames"]:
                name = f"{frame_index:03d}.jpg"
                (directory / name).write_bytes(base64.b64decode(uri.split(",", 1)[1], validate=True))
                frames.append(f"/recordings/{case}/{name}")
                frame_index += 1
            # The chart describes the candidates for the decision just taken.
            actions = steps[-1]["actions"] if steps else snapshot["actions"]
            steps.append({**snapshot, "frames": frames, "actions": actions,
                          "decision": snapshot.get("decision")})
            if snapshot["done"]:
                break
            snapshot = app.step(snapshot["revision"], model=True)
        replay = {
            "case": case, "controller": "model", "model": client.model_id, "seed": seed,
            "note": (
                f"Seed {seed}. Recorded checkpoint decisions in the native environment. Each selected action "
                "is executed unchanged. " + (
                    "The checkpoint first chooses an immediate objective, then a native action."
                    if case == "crafter" else
                    "Start in the corridor's final room, kill its left and right enemies, then advance."
                )
            ),
            "images": images, "steps": steps,
        }
        (directory / "replay.json").write_text(json.dumps(replay, indent=2) + "\n")
        target, previous = root / case, Path(staging.name) / "previous"
        if target.exists():
            target.rename(previous)
        try:
            directory.rename(target)
        except OSError:
            if previous.exists():
                previous.rename(target)
            raise
        return replay
    finally:
        app.close()
        staging.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="running JevAny model server")
    parser.add_argument("--model", required=True, help="checkpoint identity sent to the server")
    parser.add_argument("--media-root", type=Path, default=Path("/tmp/jevany-media"),
                        help="shared JEVANY_MEDIA_ROOT directory on the model server")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--out", type=Path, default=Path("jevany/demos/recordings"))
    parser.add_argument("--cases", nargs="+", choices=["crafter", "doom"], default=["crafter", "doom"])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    client = JevClient(args.base_url, timeout=args.timeout, model=args.model)
    for case in args.cases:
        replay = record(client, case, args.seed, args.out, images=not args.text_only,
                        media_root=None if args.text_only else args.media_root)
        final = replay["steps"][-1]
        print(json.dumps({"case": case, "seed": args.seed, "model": args.model,
                          "actions": len(replay["steps"]) - 1, "success": final["success"]}), flush=True)


if __name__ == "__main__":
    main()
