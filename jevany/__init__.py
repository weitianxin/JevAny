"""Train and deploy your own Jev-style decision models."""

__version__ = "0.2.0"

from .api import Choice, Noul, Score, SystemOneRequest
from .client import JevClient
from .inference import InferenceOptions

__all__ = ["Choice", "Noul", "Score", "SystemOneRequest", "JevClient", "JevModel", "InferenceOptions"]


def __getattr__(name: str):
    # HTTP clients and data tools do not need to import a model runtime.
    if name == "JevModel":
        from .runtime import JevModel
        return JevModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
