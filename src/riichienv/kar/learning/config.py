from __future__ import annotations

from dataclasses import asdict, dataclass

from ..contracts import content_hash
from ..versioning import FEATURE_CONTRACT_VERSION, LOSS_CONTRACT_VERSION

MODEL_VERSION = "kar_transformer_v2_legal_set"
FEATURE_VERSION = FEATURE_CONTRACT_VERSION
LOSS_VERSION = LOSS_CONTRACT_VERSION


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    width: int = 256
    heads: int = 4
    feedforward: int = 704
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    preset: str = "small"

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.layers, self.width, self.heads, self.feedforward)
        ):
            raise ValueError("Model dimensions must be positive integers")
        if self.width % self.heads or self.width // self.heads % 2:
            raise ValueError("RoPE requires an even integral attention head dimension")
        if self.norm_eps <= 0 or self.rope_base <= 1:
            raise ValueError("Invalid norm/RoPE constants")

    @property
    def parameter_count(self) -> int:
        return (
            self.layers * (4 * self.width**2 + 3 * self.width * self.feedforward + 2 * self.width)
            + 1636 * self.width
            + 651
        )

    @property
    def contract_hash(self) -> str:
        return content_hash({"version": MODEL_VERSION, **asdict(self)})

    @classmethod
    def from_preset(cls, name: str) -> ModelConfig:
        presets = {
            "tiny": (4, 192, 3, 512),
            "small": (6, 256, 4, 704),
            "medium": (8, 384, 6, 1024),
            "large": (8, 512, 8, 1408),
        }
        if name not in presets:
            raise ValueError("Unknown KAR model preset")
        return cls(*presets[name], preset=name)
