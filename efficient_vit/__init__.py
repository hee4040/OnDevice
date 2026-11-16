"""KD ThinkingViT with budget-aware routing and static GPU execution."""
from .policy import Policy, PRESETS
from .model import load_native, specialize
from .runtime import Engine

__all__ = ["Policy", "PRESETS", "load_native", "specialize", "Engine"]
