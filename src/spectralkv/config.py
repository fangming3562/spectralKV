"""Public, deliberately small configuration surface."""

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Config:
    percentages: tuple[int, ...] = (5, 10)
    layer_budget: str = "dynamic"
    profile: str | None = None
    layer_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if not self.percentages or len(set(self.percentages)) != len(self.percentages):
            raise ValueError("Specify nonempty, unique integer percentages.")
        if any(type(p) is not int or not 0 < p <= 100 for p in self.percentages):
            raise ValueError("Each percentage must be an integer in [1, 100].")
        if self.layer_budget not in ("dynamic", "fixed"):
            raise ValueError("layer_budget must be dynamic or fixed.")
        if self.layer_budget == "dynamic" and (self.profile is not None or self.layer_weights is not None):
            raise ValueError("Profiles and explicit weights apply only to fixed layer budgets.")
        if self.profile is not None and self.layer_weights is not None:
            raise ValueError("Choose a profile or explicit layer_weights, not both.")
        if self.layer_weights is not None and (
            not self.layer_weights or any(not math.isfinite(w) or w < 0 for w in self.layer_weights)
        ):
            raise ValueError("Layer weights must be finite and nonnegative.")

    def fixed_weights(self, model_config):
        if self.layer_weights is not None:
            weights = list(self.layer_weights)
        else:
            name = self.profile or {"llama": "llama31_8b", "qwen3": "qwen3_4b"}.get(model_config.model_type)
            if name in ("llama31_8b", "qwen3_4b"):
                text = files(__package__).joinpath("profiles", name + ".json").read_text()
            elif name:
                text = Path(name).read_text()
            else:
                raise ValueError("A matching fixed layer profile is required.")
            profile = json.loads(text)
            if any(getattr(model_config, k, None) != v for k, v in profile["architecture"].items()):
                raise ValueError("The fixed profile does not match this model architecture.")
            weights = profile["weights"]
        if len(weights) != model_config.num_hidden_layers or any(
            not math.isfinite(w) or w < 0 for w in weights
        ):
            raise ValueError("One finite, nonnegative weight per layer is required.")
        return weights
