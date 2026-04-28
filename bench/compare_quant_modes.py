#!/usr/bin/env python3
"""Compare torch-vs-FP8 matrix outputs and summarize quantization robustness gaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUT_FIELDS = [
    "train_experiment",
    "eval_config",
    "torch_mode",
    "fp8_mode",
    "torch_avg_ler_after",
    "fp8_avg_ler_after",
    "abs_ler_gap",
    "rel_ler_gap_pct",
    "torch_avg_speedup",
    "fp8_avg_speedup",
    "fp8_minus_torch_speedup",
]


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("train_experiment", ""), row.get("eval_config", ""))
        out[key] = row
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "| Train experiment | Eval config | Torch avg LER | FP8 avg LER | Abs gap | Rel gap (%) | Torch speedup | FP8 speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {train} | {cfg} | {torch_ler:.6f} | {fp8_ler:.6f} | {gap:.6f} | {rel:.2f} | {torch_spd:.3f}x | {fp8_spd:.3f}x |".format(
                train=row["train_experiment"],
                cfg=row["eval_config"],
                torch_ler=float(row["torch_avg_ler_after"]),
                fp8_ler=float(row["fp8_avg_ler_after"]),
                gap=float(row["abs_ler_gap"]),
                rel=float(row["rel_ler_gap_pct"]),
                torch_spd=float(row["torch_avg_speedup"]),
                fp8_spd=float(row["fp8_avg_speedup"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-csv", required=True)
    parser.add_argument("--fp8-csv", required=True)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    torch_csv = Path(args.torch_csv).resolve()
    fp8_csv = Path(args.fp8_csv).resolve()
    torch_rows = index_rows(load_csv(torch_csv))
    fp8_rows = index_rows(load_csv(fp8_csv))

    shared_keys = sorted(set(torch_rows) & set(fp8_rows))
    if not shared_keys:
        raise SystemExit("no overlapping successful train/eval rows between torch and fp8 inputs")

    out_rows: list[dict[str, object]] = []
    for key in shared_keys:
        torch_row = torch_rows[key]
        fp8_row = fp8_rows[key]
        torch_ler = safe_float(torch_row.get("avg_ler_after", "nan"))
        fp8_ler = safe_float(fp8_row.get("avg_ler_after", "nan"))
        torch_speedup = safe_float(torch_row.get("avg_speedup", "nan"))
        fp8_speedup = safe_float(fp8_row.get("avg_speedup", "nan"))
        abs_gap = fp8_ler - torch_ler
        rel_gap = (abs_gap / torch_ler * 100.0) if torch_ler == torch_ler and torch_ler != 0.0 else float("nan")
        out_rows.append(
            {
                "train_experiment": key[0],
                "eval_config": key[1],
                "torch_mode": torch_row.get("eval_mode", ""),
                "fp8_mode": fp8_row.get("eval_mode", ""),
                "torch_avg_ler_after": torch_ler,
                "fp8_avg_ler_after": fp8_ler,
                "abs_ler_gap": abs_gap,
                "rel_ler_gap_pct": rel_gap,
                "torch_avg_speedup": torch_speedup,
                "fp8_avg_speedup": fp8_speedup,
                "fp8_minus_torch_speedup": fp8_speedup - torch_speedup,
            }
        )

    csv_out = Path(args.csv_out).resolve() if args.csv_out else torch_csv.parent.parent / "quantization_gap.csv"
    md_out = Path(args.md_out).resolve() if args.md_out else torch_csv.parent.parent / "quantization_gap.md"
    write_csv(out_rows, csv_out)
    write_markdown(out_rows, md_out)
    print(f"[quant-gap] wrote {csv_out}")
    print(f"[quant-gap] wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
