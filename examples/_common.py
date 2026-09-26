"""Select the same HTTP or in-process backend for each example."""
import argparse

from jevany.client import DecisionClient, JevClient


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    backend = result.add_mutually_exclusive_group()
    backend.add_argument("--base-url", default="http://127.0.0.1:8008")
    backend.add_argument("--checkpoint", help="run a checkpoint in this Python process")
    result.add_argument("--device", choices=["cpu", "mps", "cuda"])
    result.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    return result


def client(args: argparse.Namespace) -> DecisionClient:
    if args.checkpoint:
        from jevany import JevModel
        return JevModel.from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    if args.device or args.dtype:
        raise ValueError("--device and --dtype require --checkpoint")
    return JevClient(args.base_url)
