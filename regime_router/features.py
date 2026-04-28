from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .catalog import load_regime_catalog, load_yaml
from .types import RegimeSpec


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den == 0.0:
        return default
    return num / den


def extract_noise_features(noise_model: dict[str, Any]) -> dict[str, float]:
    prep = [float(noise_model.get("p_prep_X", 0.0)), float(noise_model.get("p_prep_Z", 0.0))]
    meas = [float(noise_model.get("p_meas_X", 0.0)), float(noise_model.get("p_meas_Z", 0.0))]
    idle_cnot_x = float(noise_model.get("p_idle_cnot_X", 0.0))
    idle_cnot_y = float(noise_model.get("p_idle_cnot_Y", 0.0))
    idle_cnot_z = float(noise_model.get("p_idle_cnot_Z", 0.0))
    idle_spam_x = float(noise_model.get("p_idle_spam_X", 0.0))
    idle_spam_y = float(noise_model.get("p_idle_spam_Y", 0.0))
    idle_spam_z = float(noise_model.get("p_idle_spam_Z", 0.0))

    cnot_pairs = {k: float(v) for k, v in noise_model.items() if k.startswith("p_cnot_")}
    cnot_total = sum(cnot_pairs.values())
    cnot_z = sum(v for k, v in cnot_pairs.items() if "Z" in k)
    cnot_x = sum(v for k, v in cnot_pairs.items() if "X" in k)
    cnot_two_body = sum(v for k, v in cnot_pairs.items() if len(k) == len("p_cnot_XX"))
    idle_cnot_total = idle_cnot_x + idle_cnot_y + idle_cnot_z
    idle_spam_total = idle_spam_x + idle_spam_y + idle_spam_z

    features = {
        "prep_mean": _mean(prep),
        "prep_asymmetry": abs(prep[0] - prep[1]),
        "meas_mean": _mean(meas),
        "meas_asymmetry": abs(meas[0] - meas[1]),
        "idle_cnot_total": idle_cnot_total,
        "idle_cnot_z_fraction": _safe_ratio(idle_cnot_z, idle_cnot_total),
        "idle_cnot_z_to_x": _safe_ratio(idle_cnot_z, max(idle_cnot_x, 1e-12)),
        "idle_spam_total": idle_spam_total,
        "idle_spam_z_fraction": _safe_ratio(idle_spam_z, idle_spam_total),
        "idle_spam_z_to_x": _safe_ratio(idle_spam_z, max(idle_spam_x, 1e-12)),
        "cnot_total": cnot_total,
        "cnot_z_fraction": _safe_ratio(cnot_z, cnot_total),
        "cnot_x_fraction": _safe_ratio(cnot_x, cnot_total),
        "cnot_two_body_fraction": _safe_ratio(cnot_two_body, cnot_total),
        "cnot_z_to_x": _safe_ratio(cnot_z, max(cnot_x, 1e-12)),
    }
    return features


def _apply_relative_jitter(features: dict[str, float], rng: random.Random, scale: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in features.items():
        if value == 0.0:
            jittered = abs(rng.gauss(0.0, scale * 0.01))
        else:
            jittered = value * (1.0 + rng.gauss(0.0, scale))
        if "fraction" in name:
            jittered = min(max(jittered, 0.0), 1.0)
        else:
            jittered = max(jittered, 0.0)
        out[name] = jittered
    return out


def make_proxy_window_features(
    base_features: dict[str, float],
    *,
    rng: random.Random,
    noise_scale: float,
    window_size_shots: int,
) -> dict[str, float]:
    out = _apply_relative_jitter(base_features, rng, noise_scale)
    detector_rate = min(max(
        0.015
        + 2.0 * out["meas_mean"]
        + 7.0 * out["idle_cnot_total"]
        + 3.0 * out["cnot_total"],
        0.0,
    ), 1.0)
    x_like_rate = detector_rate / max(1.0 + out["idle_cnot_z_to_x"], 1e-9)
    z_like_rate = detector_rate - x_like_rate
    residual_weight_mean = 4.0 + 60.0 * out["cnot_total"] + 12.0 * out["idle_spam_total"]
    temporal_corr = min(max(0.05 + 0.15 * out["idle_spam_z_fraction"] + rng.gauss(0.0, noise_scale * 0.2), 0.0), 1.0)
    burstiness = min(max(0.05 + 0.2 * out["meas_asymmetry"] + rng.gauss(0.0, noise_scale * 0.2), 0.0), 1.0)

    out.update(
        {
            "window_size_shots": float(window_size_shots),
            "detector_rate_mean": detector_rate,
            "detector_rate_std": max(0.0, detector_rate * (0.08 + noise_scale)),
            "x_like_trigger_rate": max(x_like_rate, 0.0),
            "z_like_trigger_rate": max(z_like_rate, 0.0),
            "temporal_corr_lag1": temporal_corr,
            "burstiness": burstiness,
            "residual_weight_mean": residual_weight_mean,
            "residual_weight_std": max(0.1, math.sqrt(max(residual_weight_mean, 0.1))),
        }
    )
    return out


def load_regime_base_features(
    *,
    repo_root: Path,
    catalog_path: Path,
) -> dict[str, dict[str, float]]:
    catalog = load_regime_catalog(catalog_path)
    config_dir = repo_root / "conf"
    out = {}
    for name, spec in catalog.items():
        config_path = config_dir / f"{spec.config_name}.yaml"
        cfg = load_yaml(config_path)
        noise_model = dict(cfg.get("data", {}).get("noise_model", {}))
        out[name] = extract_noise_features(noise_model)
    return out
