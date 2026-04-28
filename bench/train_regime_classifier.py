#!/usr/bin/env python3
"""Train the regime router classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_router.catalog import default_router_config_path, load_router_config
from regime_router.classifier import (
    load_dataset_csv,
    markdown_report,
    save_model,
    train_softmax_classifier,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--router-config", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    router_config_path = Path(args.router_config).resolve() if args.router_config else default_router_config_path(repo_root)
    cfg = load_router_config(router_config_path)
    train_cfg = cfg["training"]

    dataset_path = Path(args.dataset).resolve() if args.dataset else repo_root / "results" / "router" / "datasets" / "regime_router_dataset.csv"
    model_out = Path(args.model_out).resolve() if args.model_out else repo_root / "results" / "router" / "models" / "regime_router_softmax.json"
    report_out = Path(args.report_out).resolve() if args.report_out else repo_root / "results" / "router" / "models" / "regime_router_softmax.md"

    rows, feature_names = load_dataset_csv(dataset_path)
    model = train_softmax_classifier(
        rows,
        feature_names,
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        l2=float(train_cfg["l2"]),
        seed=int(train_cfg["seed"]),
    )
    save_model(model, model_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(markdown_report(model))
    print(f"[router-model] wrote {model_out}")
    print(f"[router-model] wrote {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
