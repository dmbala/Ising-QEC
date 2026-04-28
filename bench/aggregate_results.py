#!/usr/bin/env python3
"""Aggregate matrix result JSON files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "train_experiment",
    "eval_config",
    "eval_mode",
    "status",
    "avg_ler_baseline",
    "avg_ler_after",
    "avg_ler_delta",
    "avg_ler_ratio",
    "avg_latency_baseline_us",
    "avg_latency_after_us",
    "avg_speedup",
]


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def find_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.json"))


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "train_experiment" not in payload or "eval_config" not in payload:
            continue
        avg_base = payload.get("avg_ler_baseline")
        avg_after = payload.get("avg_ler_after")
        if isinstance(avg_base, (int, float)) and isinstance(avg_after, (int, float)):
            payload["avg_ler_delta"] = avg_after - avg_base
            payload["avg_ler_ratio"] = (avg_after / avg_base) if avg_base else float("inf")
        rows.append(payload)
    rows.sort(key=lambda row: (str(row.get("train_experiment")), str(row.get("eval_config"))))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "| Train experiment | Eval config | Status | Avg LER (baseline) | Avg LER (after) | Delta | Ratio | Avg latency baseline (µs) | Avg latency after (µs) | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {train} | {eval_cfg} | {status} | {base:.6f} | {after:.6f} | {delta:.6f} | {ratio:.3f} | {lat_base:.3f} | {lat_after:.3f} | {speedup:.3f}x |".format(
                train=row.get("train_experiment", ""),
                eval_cfg=row.get("eval_config", ""),
                status=row.get("status", ""),
                base=safe_float(row.get("avg_ler_baseline")),
                after=safe_float(row.get("avg_ler_after")),
                delta=safe_float(row.get("avg_ler_delta")),
                ratio=safe_float(row.get("avg_ler_ratio")),
                lat_base=safe_float(row.get("avg_latency_baseline_us")),
                lat_after=safe_float(row.get("avg_latency_after_us")),
                speedup=safe_float(row.get("avg_speedup")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Directory containing result JSON files or one JSON file.")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    rows = load_rows(find_json_files(input_path))
    if not rows:
        raise SystemExit(f"no matrix result JSON files found under {input_path}")

    default_parent = input_path if input_path.is_dir() else input_path.parent
    csv_out = Path(args.csv_out).resolve() if args.csv_out else default_parent / "aggregate.csv"
    md_out = Path(args.md_out).resolve() if args.md_out else default_parent / "aggregate.md"
    write_csv(rows, csv_out)
    write_markdown(rows, md_out)
    print(f"[aggregate] wrote {csv_out}")
    print(f"[aggregate] wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
