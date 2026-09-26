"""The `jevany` command: train, serve, decide, explore demos, and prepare data."""
import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="jevany", description="Train and deploy Jev-style decision models.")
    from . import __version__
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=["train", "serve", "decide", "data", "demo"])
    if not argv or argv[0] in ("-h", "--help", "--version"):
        parser.parse_args(argv or ["--help"])
        return
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    try:
        if command == "train":
            from .train import main as train
            train(rest)
        elif command == "serve":
            from .serve import main as serve
            serve(rest)
        elif command == "data":
            data_main(rest)
        elif command == "demo":
            from .demos.server import main as demo
            demo(rest)
        else:
            decide_main(rest)
    except ImportError as error:
        parser.exit(2, f"{error}\nInstall the needed extra: 'jevany[train]', 'jevany[serve]' or 'jevany[local]'.\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"jevany: {error}\n")


def decide_main(argv: list[str]) -> None:
    from dataclasses import fields
    from .client import JevClient
    from .inference import InferenceOptions, add_inference_arguments, inference_options_from_args

    parser = argparse.ArgumentParser(prog="jevany decide")
    parser.add_argument("request", help="JSON request file; - reads stdin")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", help="load this checkpoint in-process")
    mode.add_argument("--base-url", default="http://127.0.0.1:8008", help="call a running server")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model-name", help="identity for a locally loaded checkpoint")
    add_inference_arguments(parser)
    args = parser.parse_args(argv)
    content = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
    from .api import SystemOneRequest
    request = SystemOneRequest.model_validate_json(content)
    if args.checkpoint:
        from .runtime import JevModel
        client = JevModel.from_pretrained(
            args.checkpoint, device=args.device, dtype=args.dtype, model_name=args.model_name,
            inference_options=inference_options_from_args(args),
        )
    else:
        if (args.device or args.dtype or args.model_name is not None
                or any(getattr(args, item.name) is not None for item in fields(InferenceOptions))):
            parser.error("device, dtype, model-name and inference limit options require --checkpoint")
        client = JevClient(args.base_url)
    print(json.dumps(client(request), indent=2, ensure_ascii=False))


def data_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="jevany data", description="Prepare and validate labelled System One JSONL.")
    parser.add_argument("action", choices=["init", "validate", "build-sft", "build-rlcr"])
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv or ["--help"])
        return
    action = parser.parse_args(argv[:1]).action
    rest = argv[1:]
    if action == "build-sft":
        from .datasets.build_sft import main as build
        build(rest)
    elif action == "build-rlcr":
        from .datasets.build_rlcr import main as build
        build(rest)
    else:
        sub = argparse.ArgumentParser(prog=f"jevany data {action}")
        if action == "init":
            sub.add_argument("--out", default="data/starter")
            from .datasets import init_starter
            print(init_starter(sub.parse_args(rest).out))
        else:
            sub.add_argument("path")
            from .data import validate_dataset
            print(json.dumps(validate_dataset(sub.parse_args(rest).path), indent=2))
