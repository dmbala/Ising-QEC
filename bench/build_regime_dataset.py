#!/usr/bin/env python3
"""Build a proxy routing dataset for the regime router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_router.catalog import default_catalog_path, default_router_config_path, load_router_config
from regime_router.dataset import build_proxy_dataset, write_dataset_csv, write_dataset_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--catalog", default="")
    parser.add_argument("--router-config", default="")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    catalog_path = Path(args.catalog).resolve() if args.catalog else default_catalog_path(repo_root)
    router_config_path = Path(args.router_config).resolve() if args.router_config else default_router_config_path(repo_root)
    cfg = load_router_config(router_config_path)
    ds_cfg = cfg["dataset"]

    rows = build_proxy_dataset(
        repo_root=repo_root,
        catalog_path=catalog_path,
        num_windows_per_regime=int(ds_cfg["num_windows_per_regime"]),
        noise_scale=float(ds_cfg["noise_scale"]),
        window_size_shots=int(ds_cfg["window_size_shots"]),
        train_fraction=float(ds_cfg["splits"]["train"]),
        valid_fraction=float(ds_cfg["splits"]["valid"]),
        seed=int(ds_cfg["seed"]),
    )

    dataset_root = repo_root / "results" / "router" / "datasets"
    csv_out = Path(args.csv_out).resolve() if args.csv_out else dataset_root / "regime_router_dataset.csv"
    jsonl_out = Path(args.jsonl_out).resolve() if args.jsonl_out else dataset_root / "regime_router_dataset.jsonl"
    metadata_out = dataset_root / "regime_router_dataset.meta.json"

    write_dataset_csv(rows, csv_out)
    write_dataset_jsonl(rows, jsonl_out)
    metadata_out.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "catalog_path": str(catalog_path),
                "router_config_path": str(router_config_path),
                "csv_out": str(csv_out),
                "jsonl_out": str(jsonl_out),
            },
            indent=2,
        )
    )
    print(f"[router-dataset] wrote {csv_out}")
    print(f"[router-dataset] wrote {jsonl_out}")
    print(f"[router-dataset] wrote {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
