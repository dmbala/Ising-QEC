from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_dataset_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    feature_names = [
        key for key in rows[0].keys()
        if key not in {"regime_name", "split", "window_id", "source"}
    ]
    return rows, feature_names


def rows_to_arrays(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    split: str,
) -> tuple[list[list[float]], list[int], list[str]]:
    filtered = [row for row in rows if row.get("split") == split]
    if not filtered:
        raise ValueError(f"no rows for split={split!r}")
    classes = sorted({str(row["regime_name"]) for row in rows})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    x = [[float(row[name]) for name in feature_names] for row in filtered]
    y = [class_to_idx[str(row["regime_name"])] for row in filtered]
    return x, y, classes


def _column_stats(x: list[list[float]]) -> tuple[list[float], list[float]]:
    cols = len(x[0])
    mean = [0.0] * cols
    std = [0.0] * cols
    n = float(len(x))
    for row in x:
        for idx, value in enumerate(row):
            mean[idx] += value
    mean = [value / n for value in mean]
    for row in x:
        for idx, value in enumerate(row):
            std[idx] += (value - mean[idx]) ** 2
    std = [math.sqrt(value / n) if value > 0.0 else 1.0 for value in std]
    std = [value if value != 0.0 else 1.0 for value in std]
    return mean, std


def _standardize_rows(x: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    out = []
    for row in x:
        out.append([(value - mean[idx]) / std[idx] for idx, value in enumerate(row)])
    return out


def _softmax(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(value - mx) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _predict_logits(weights: list[list[float]], bias: list[float], row: list[float]) -> list[float]:
    logits = []
    for class_idx in range(len(bias)):
        total = bias[class_idx]
        for feat_idx, feat_value in enumerate(row):
            total += weights[feat_idx][class_idx] * feat_value
        logits.append(total)
    return logits


def _evaluate_internal(
    weights: list[list[float]],
    bias: list[float],
    mean: list[float],
    std: list[float],
    x: list[list[float]],
    y: list[int],
    classes: list[str],
) -> dict[str, Any]:
    xs = _standardize_rows(x, mean, std)
    preds: list[int] = []
    for row in xs:
        probs = _softmax(_predict_logits(weights, bias, row))
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        preds.append(pred_idx)
    correct = sum(int(pred == truth) for pred, truth in zip(preds, y))
    confusion: dict[str, dict[str, int]] = {}
    for true_idx, true_name in enumerate(classes):
        row_counts = {}
        pred_counts = Counter(pred for pred, truth in zip(preds, y) if truth == true_idx)
        for pred_idx, pred_name in enumerate(classes):
            row_counts[pred_name] = int(pred_counts.get(pred_idx, 0))
        confusion[true_name] = row_counts
    return {"accuracy": correct / max(len(y), 1), "confusion": confusion}


def train_softmax_classifier(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    x_train, y_train, classes = rows_to_arrays(rows, feature_names, "train")
    x_valid, y_valid, _ = rows_to_arrays(rows, feature_names, "valid")
    mean, std = _column_stats(x_train)
    x_train_std = _standardize_rows(x_train, mean, std)
    x_valid_std = _standardize_rows(x_valid, mean, std)

    rng = random.Random(seed)
    weights = [[rng.gauss(0.0, 0.01) for _ in classes] for _ in feature_names]
    bias = [0.0 for _ in classes]

    best_state = None
    best_valid = float("inf")
    best_epoch = 0
    class_count = len(classes)

    for epoch in range(epochs):
        grad_w = [[0.0 for _ in classes] for _ in feature_names]
        grad_b = [0.0 for _ in classes]
        for row, truth in zip(x_train_std, y_train):
            probs = _softmax(_predict_logits(weights, bias, row))
            for class_idx in range(class_count):
                error = probs[class_idx] - (1.0 if class_idx == truth else 0.0)
                grad_b[class_idx] += error
                for feat_idx, feat_value in enumerate(row):
                    grad_w[feat_idx][class_idx] += feat_value * error
        scale = 1.0 / max(len(x_train_std), 1)
        for feat_idx in range(len(feature_names)):
            for class_idx in range(class_count):
                grad = grad_w[feat_idx][class_idx] * scale + l2 * weights[feat_idx][class_idx]
                weights[feat_idx][class_idx] -= learning_rate * grad
        for class_idx in range(class_count):
            bias[class_idx] -= learning_rate * grad_b[class_idx] * scale

        valid_loss = 0.0
        for row, truth in zip(x_valid_std, y_valid):
            probs = _softmax(_predict_logits(weights, bias, row))
            valid_loss += -math.log(max(probs[truth], 1e-12))
        valid_loss /= max(len(x_valid_std), 1)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = (
                [col[:] for col in weights],
                bias[:],
            )

    assert best_state is not None
    weights, bias = best_state
    train_metrics = _evaluate_internal(weights, bias, mean, std, x_train, y_train, classes)
    valid_metrics = _evaluate_internal(weights, bias, mean, std, x_valid, y_valid, classes)
    return {
        "model_type": "softmax",
        "feature_names": feature_names,
        "classes": classes,
        "mean": mean,
        "std": std,
        "weights": weights,
        "bias": bias,
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid,
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
        },
    }


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_model(model: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def predict_proba(model: dict[str, Any], features: dict[str, float]) -> dict[str, float]:
    feature_names = list(model["feature_names"])
    mean = list(model["mean"])
    std = list(model["std"])
    row = [(float(features[name]) - mean[idx]) / std[idx] for idx, name in enumerate(feature_names)]
    probs = _softmax(_predict_logits(model["weights"], model["bias"], row))
    return {name: float(prob) for name, prob in zip(model["classes"], probs)}


def predict_regime(model: dict[str, Any], features: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    probs = predict_proba(model, features)
    best_name = max(probs, key=probs.get)
    return best_name, probs[best_name], probs


def evaluate_saved_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    feature_names = list(model["feature_names"])
    filtered = [row for row in rows if row.get("split") == split]
    if not filtered:
        raise ValueError(f"no rows for split={split!r}")
    classes = list(model["classes"])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    x = [[float(row[name]) for name in feature_names] for row in filtered]
    y = [class_to_idx[str(row["regime_name"])] for row in filtered]
    return _evaluate_internal(model["weights"], model["bias"], model["mean"], model["std"], x, y, classes)


def markdown_report(model: dict[str, Any]) -> str:
    train = model["metrics"]["train"]
    valid = model["metrics"]["valid"]
    return "\n".join(
        [
            "# Regime Router Classifier",
            "",
            f"- model: `{model['model_type']}`",
            f"- classes: `{', '.join(model['classes'])}`",
            f"- features: `{len(model['feature_names'])}`",
            f"- train accuracy: `{train['accuracy']:.4f}`",
            f"- valid accuracy: `{valid['accuracy']:.4f}`",
            f"- best epoch: `{model['training']['best_epoch']}`",
            f"- best valid loss: `{model['training']['best_valid_loss']:.6f}`",
            "",
            "## Valid confusion",
            "",
            "```json",
            json.dumps(valid["confusion"], indent=2),
            "```",
            "",
        ]
    )
