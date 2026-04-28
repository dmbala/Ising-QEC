from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegimeSpec:
    name: str
    config_name: str
    experiment_name: str
    description: str = ""
    default: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureWindow:
    regime_name: str
    split: str
    window_id: str
    source: str
    features: dict[str, float]


@dataclass
class RouterExample:
    regime_name: str
    split: str
    features: dict[str, float]
    source: str = "config_proxy"
    window_id: str = ""


@dataclass
class RouterDecision:
    predicted_regime: str
    selected_regime: str
    confidence: float
    used_fallback: bool
    probabilities: dict[str, float]
