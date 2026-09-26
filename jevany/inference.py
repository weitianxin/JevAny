"""Inference limits shared by Python, the CLI and HTTP deployments."""
import argparse
import os
from dataclasses import dataclass, fields, replace
from typing import Mapping


@dataclass(frozen=True)
class InferenceOptions:
    """Admission and cache limits for one loaded model.

    Token limits reject oversized inputs rather than truncating them. State and
    branch limits are also bounded by the backbone's context window. A branch
    includes the state; the packed limit covers the entire encoded request.
    Set prefix_cache_size=0 to disable prefix reuse.
    """

    max_state_tokens: int = 8192
    max_branch_tokens: int = 8192
    max_packed_tokens: int = 8192
    prefix_cache_size: int = 4
    prefix_min_tokens: int = 384

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            minimum = 0 if item.name.startswith("prefix_") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{item.name} must be an integer >= {minimum}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InferenceOptions":
        """Read JEVANY_MAX_*_TOKENS and the existing JEVANY_PREFIX_* settings."""
        env = os.environ if env is None else env
        names = {item.name: "JEVANY_" + item.name.upper() for item in fields(cls)}
        names["prefix_cache_size"] = "JEVANY_PREFIX_CACHE"
        settings = {}
        for field_name, variable in names.items():
            if variable in env:
                try:
                    settings[field_name] = int(env[variable])
                except ValueError as error:
                    raise ValueError(f"{variable} must be an integer") from error
        return cls(**settings)


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same optional limit overrides to serve and local decide."""
    for item in fields(InferenceOptions):
        parser.add_argument("--" + item.name.replace("_", "-"), type=int, default=None,
                            help=f"override {item.name}; otherwise use JEVANY_* or runtime defaults")


def inference_options_from_args(args: argparse.Namespace) -> InferenceOptions:
    """CLI overrides take precedence over environment settings."""
    overrides = {item.name: getattr(args, item.name) for item in fields(InferenceOptions)
                 if getattr(args, item.name) is not None}
    return replace(InferenceOptions.from_env(), **overrides)
