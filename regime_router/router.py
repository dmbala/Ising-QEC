from __future__ import annotations

from .classifier import predict_regime
from .types import RouterDecision


def route_features(
    *,
    model: dict,
    features: dict[str, float],
    default_regime: str,
    confidence_threshold: float,
) -> RouterDecision:
    predicted, confidence, probabilities = predict_regime(model, features)
    used_fallback = confidence < confidence_threshold
    selected = default_regime if used_fallback else predicted
    return RouterDecision(
        predicted_regime=predicted,
        selected_regime=selected,
        confidence=confidence,
        used_fallback=used_fallback,
        probabilities=probabilities,
    )
