from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .catalog import load_regime_catalog
from .features import load_regime_base_features, make_proxy_window_features
from .types import RouterExample


def _split_name(index: int, total: int, train_fraction: float, valid_fraction: float) -> str:
    train_cut = int(total * train_fraction)
    valid_cut = int(total * (train_fraction + valid_fraction))
    if index < train_cut:
        return "train"
    if index < valid_cut:
        return "valid"
    return "test"


def build_proxy_dataset(
    *,
    repo_root: Path,
    catalog_path: Path,
    num_windows_per_regime: int,
    noise_scale: float,
    window_size_shots: int,
    train_fraction: float,
    valid_fraction: float,
    seed: int,
) -> list[RouterExample]:
    base_features = load_regime_base_features(repo_root=repo_root, catalog_path=catalog_path)
    catalog = load_regime_catalog(catalog_path)
    rng = random.Random(seed)
    rows: list[RouterExample] = []
    for regime_name in catalog:
        for index in range(num_windows_per_regime):
            features = make_proxy_window_features(
                base_features[regime_name],
                rng=rng,
                noise_scale=noise_scale,
                window_size_shots=window_size_shots,
            )
            rows.append(
                RouterExample(
                    regime_name=regime_name,
                    split=_split_name(index, num_windows_per_regime, train_fraction, valid_fraction),
                    window_id=f"{regime_name}-{index:05d}",
                    source="config_proxy",
                    features=features,
                )
            )
    return rows


def dataset_feature_names(rows: list[RouterExample]) -> list[str]:
    names = sorted({name for row in rows for name in row.features})
    return names


def write_dataset_csv(rows: list[RouterExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = dataset_feature_names(rows)
    fields = ["regime_name", "split", "window_id", "source", *feature_names]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {
                "regime_name": row.regime_name,
                "split": row.split,
                "window_id": row.window_id,
                "source": row.source,
            }
            payload.update(row.features)
            writer.writerow(payload)


def write_dataset_jsonl(rows: list[RouterExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "regime_name": row.regime_name,
                        "split": row.split,
                        "window_id": row.window_id,
                        "source": row.source,
                        "features": row.features,
                    }
                )
                + "\n"
            )
