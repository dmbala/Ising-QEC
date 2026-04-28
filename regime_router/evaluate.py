from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .catalog import choose_default_regime, load_regime_catalog, load_router_config
from .classifier import load_dataset_csv, load_model
from .router import route_features


def load_matrix_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        out[(row["train_experiment"], row["eval_config"])] = row
    if not out:
        raise ValueError(f"no successful matrix rows in {path}")
    return out


def _safe_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def evaluate_routing(
    *,
    model_path: Path,
    dataset_path: Path,
    matrix_csv: Path,
    catalog_path: Path,
    router_config_path: Path,
    split: str,
) -> dict[str, Any]:
    model = load_model(model_path)
    rows, feature_names = load_dataset_csv(dataset_path)
    filtered = [row for row in rows if row.get("split") == split]
    if not filtered:
        raise ValueError(f"no rows for split={split!r}")

    catalog = load_regime_catalog(catalog_path)
    router_cfg = load_router_config(router_config_path)
    default_regime = choose_default_regime(catalog, router_cfg)
    threshold = float(router_cfg.get("confidence_threshold", 0.7))
    matrix = load_matrix_csv(matrix_csv)

    counts = Counter()
    per_regime: dict[str, dict[str, float]] = defaultdict(lambda: {
        "count": 0.0,
        "accuracy": 0.0,
        "routed_ler": 0.0,
        "oracle_ler": 0.0,
        "default_ler": 0.0,
    })
    routed_ler = oracle_ler = default_ler = 0.0
    routed_speedup = oracle_speedup = default_speedup = 0.0

    for row in filtered:
        regime_name = str(row["regime_name"])
        features = {name: float(row[name]) for name in feature_names}
        decision = route_features(
            model=model,
            features=features,
            default_regime=default_regime,
            confidence_threshold=threshold,
        )
        true_spec = catalog[regime_name]
        chosen_spec = catalog[decision.selected_regime]
        default_spec = catalog[default_regime]

        routed_metrics = matrix[(chosen_spec.experiment_name, true_spec.config_name)]
        oracle_metrics = matrix[(true_spec.experiment_name, true_spec.config_name)]
        default_metrics = matrix[(default_spec.experiment_name, true_spec.config_name)]

        routed_ler += _safe_float(routed_metrics.get("avg_ler_after"))
        oracle_ler += _safe_float(oracle_metrics.get("avg_ler_after"))
        default_ler += _safe_float(default_metrics.get("avg_ler_after"))
        routed_speedup += _safe_float(routed_metrics.get("avg_speedup"))
        oracle_speedup += _safe_float(oracle_metrics.get("avg_speedup"))
        default_speedup += _safe_float(default_metrics.get("avg_speedup"))

        counts["examples"] += 1
        counts["correct_prediction"] += int(decision.predicted_regime == regime_name)
        counts["fallbacks"] += int(decision.used_fallback)
        counts["default_selected"] += int(decision.selected_regime == default_regime)

        bucket = per_regime[regime_name]
        bucket["count"] += 1.0
        bucket["accuracy"] += float(decision.predicted_regime == regime_name)
        bucket["routed_ler"] += _safe_float(routed_metrics.get("avg_ler_after"))
        bucket["oracle_ler"] += _safe_float(oracle_metrics.get("avg_ler_after"))
        bucket["default_ler"] += _safe_float(default_metrics.get("avg_ler_after"))

    total = max(counts["examples"], 1)
    summary = {
        "split": split,
        "default_regime": default_regime,
        "confidence_threshold": threshold,
        "examples": counts["examples"],
        "classification_accuracy": counts["correct_prediction"] / total,
        "fallback_rate": counts["fallbacks"] / total,
        "default_selection_rate": counts["default_selected"] / total,
        "avg_ler": {
            "routed": routed_ler / total,
            "oracle": oracle_ler / total,
            "default": default_ler / total,
        },
        "avg_speedup": {
            "routed": routed_speedup / total,
            "oracle": oracle_speedup / total,
            "default": default_speedup / total,
        },
        "per_regime": {},
        "matrix_csv": str(matrix_csv),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
    }
    for regime_name, bucket in sorted(per_regime.items()):
        count = max(bucket["count"], 1.0)
        summary["per_regime"][regime_name] = {
            "count": int(bucket["count"]),
            "classification_accuracy": bucket["accuracy"] / count,
            "avg_ler": {
                "routed": bucket["routed_ler"] / count,
                "oracle": bucket["oracle_ler"] / count,
                "default": bucket["default_ler"] / count,
            },
        }
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Routed Evaluation",
        "",
        f"- split: `{summary['split']}`",
        f"- default regime: `{summary['default_regime']}`",
        f"- confidence threshold: `{summary['confidence_threshold']:.2f}`",
        f"- examples: `{summary['examples']}`",
        f"- classification accuracy: `{summary['classification_accuracy']:.4f}`",
        f"- fallback rate: `{summary['fallback_rate']:.4f}`",
        "",
        "## Average metrics",
        "",
        f"- routed LER: `{summary['avg_ler']['routed']:.6f}`",
        f"- oracle LER: `{summary['avg_ler']['oracle']:.6f}`",
        f"- default LER: `{summary['avg_ler']['default']:.6f}`",
        f"- routed speedup: `{summary['avg_speedup']['routed']:.3f}x`",
        f"- oracle speedup: `{summary['avg_speedup']['oracle']:.3f}x`",
        f"- default speedup: `{summary['avg_speedup']['default']:.3f}x`",
        "",
        "## Per-regime summary",
        "",
        "| Regime | Count | Classifier acc | Routed LER | Oracle LER | Default LER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime_name, bucket in summary["per_regime"].items():
        lines.append(
            "| {name} | {count} | {acc:.4f} | {routed:.6f} | {oracle:.6f} | {default:.6f} |".format(
                name=regime_name,
                count=bucket["count"],
                acc=bucket["classification_accuracy"],
                routed=bucket["avg_ler"]["routed"],
                oracle=bucket["avg_ler"]["oracle"],
                default=bucket["avg_ler"]["default"],
            )
        )
    lines.extend(["", "```json", json.dumps(summary, indent=2), "```", ""])
    return "\n".join(lines)
