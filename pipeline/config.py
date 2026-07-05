"""Central pipeline configuration.

Everything toggleable lives here so preprocessing / windowing choices are
config-driven, not hardcoded. Load defaults, override via JSON file or
keyword arguments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class HampelConfig:
    enabled: bool = True
    window_size: int = 7      # samples on each side
    n_sigmas: float = 3.0


@dataclass(frozen=True)
class PCAConfig:
    enabled: bool = False     # optional denoiser, off by default
    n_components: int = 5     # components kept when reconstructing


@dataclass(frozen=True)
class WaveletConfig:
    enabled: bool = False     # optional denoiser, off by default
    wavelet: str = "db4"
    level: int = 3
    mode: str = "soft"


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window parameters for feature extraction."""

    window_seconds: float = 2.0
    hop_seconds: float = 0.5
    sample_rate_hz: float = 50.0  # assumed CSI packet rate; measured later on hardware

    @property
    def window_samples(self) -> int:
        return max(1, int(round(self.window_seconds * self.sample_rate_hz)))

    @property
    def hop_samples(self) -> int:
        return max(1, int(round(self.hop_seconds * self.sample_rate_hz)))


@dataclass(frozen=True)
class PipelineConfig:
    hampel: HampelConfig = field(default_factory=HampelConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    # Subcarrier selection: guard/null slots at buffer edges carry no signal.
    drop_null_subcarriers: bool = True

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            hampel=HampelConfig(**data.get("hampel", {})),
            pca=PCAConfig(**data.get("pca", {})),
            wavelet=WaveletConfig(**data.get("wavelet", {})),
            window=WindowConfig(**data.get("window", {})),
            drop_null_subcarriers=data.get("drop_null_subcarriers", True),
        )

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return replace(self, **kwargs)


DEFAULT_CONFIG = PipelineConfig()
