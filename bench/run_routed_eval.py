#!/usr/bin/env python3
"""Evaluate routed expert selection against an existing matrix CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_router.catalog import default_catalog_path, default_router_config_path
from regime_router.evaluate import evaluate_routing, markdown_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--catalog", default="")
    parser.add_argument("--router-config", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--matrix-csv", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    catalog_path = Path(args.catalog).resolve() if args.catalog else default_catalog_path(repo_root)
    router_config_path = Path(args.router_config).resolve() if args.router_config else default_router_config_path(repo_root)
    dataset_path = Path(args.dataset).resolve() if args.dataset else repo_root / "results" / "router" / "datasets" / "regime_router_dataset.csv"
    model_path = Path(args.model).resolve() if args.model else repo_root / "results" / "router" / "models" / "regime_router_softmax.json"
    split = args.split or "test"
    json_out = Path(args.json_out).resolve() if args.json_out else repo_root / "results" / "router" / "evals" / "routed_eval.json"
    md_out = Path(args.md_out).resolve() if args.md_out else repo_root / "results" / "router" / "evals" / "routed_eval.md"

    summary = evaluate_routing(
        model_path=model_path,
        dataset_path=dataset_path,
        matrix_csv=Path(args.matrix_csv).resolve(),
        catalog_path=catalog_path,
        router_config_path=router_config_path,
        split=split,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2))
    md_out.write_text(markdown_report(summary))
    print(f"[routed-eval] wrote {json_out}")
    print(f"[routed-eval] wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
