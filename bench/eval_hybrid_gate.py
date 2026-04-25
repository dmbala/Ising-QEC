#!/usr/bin/env python3
"""Estimate residual-weight-gated hybrid decoder policies from ablation logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


DECODER_TO_TIMING_KEY = {
    "No-op": None,
    "Union-Find": "uf_decode",
    "BP-only": "bp_only_decode",
    "BP+LSD-0": "bplsd_decode",
    "Uncorr-PM": "uncorr_pm",
    "Corr-PM": "corr_pm",
}
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_ablation_log(path: Path) -> dict[str, object]:
    text = path.read_text()

    basis_match = re.search(r"DECODER ABLATION STUDY\s+\|\s+basis=([A-Z])", text)
    total_match = re.search(r"Total samples:\s+(\d+)", text)
    if not basis_match or not total_match:
        raise ValueError(f"could not parse basis/total samples from {path}")
    basis = basis_match.group(1)
    total_samples = int(total_match.group(1))

    timing = {}
    timing_block = re.search(r"TIMING BREAKDOWN.*?={20,}\n(.*?)\n={20,}", text, flags=re.S)
    if timing_block:
        for line in timing_block.group(1).splitlines():
            match = re.match(r"\s*([A-Za-z0-9_+\-]+)\s+([0-9.]+)s", line)
            if match:
                timing[match.group(1)] = float(match.group(2))

    overall_ler = {}
    logical_block = re.search(r"--- Logical Error Rates ---\n(.*?)(?:\n--- |\n={20,})", text, flags=re.S)
    if logical_block:
        for line in logical_block.group(1).splitlines():
            if "Baseline (no pre-dec)" in line:
                match = re.search(r"LER = (" + FLOAT_RE + r")", line)
                if match:
                    overall_ler["Baseline (no pre-dec)"] = float(match.group(1))
                continue
            match = re.match(r"\s*([A-Za-z0-9+\-]+)\s+LER = (" + FLOAT_RE + r")", line)
            if match:
                overall_ler[match.group(1)] = float(match.group(2))

    weight_counts: dict[int, int] = {}
    dist_block = re.search(r"--- Residual Weight Distribution ---\n(.*?)(?:\n--- |\n={20,})", text, flags=re.S)
    if dist_block:
        for line in dist_block.group(1).splitlines():
            match = re.match(r"\s*Weight\s+(\d+\+?|\d+):\s+(\d+)", line)
            if match:
                label = match.group(1)
                bucket = int(label.rstrip("+"))
                weight_counts[bucket] = int(match.group(2))

    conditional_block = re.search(r"--- Conditional LER by Residual Weight ---\n(.*?)(?:\n={20,}|\Z)", text, flags=re.S)
    conditional_ler: dict[int, dict[str, float]] = {}
    if conditional_block:
        lines = [line.rstrip() for line in conditional_block.group(1).splitlines() if line.strip()]
        header = None
        for line in lines:
            columns = re.split(r"\s{2,}", line.strip())
            if not columns:
                continue
            if columns[0] == "Weight":
                header = columns
                continue
            if header is None or line.startswith("="):
                continue
            label = columns[0]
            bucket = int(label.rstrip("+"))
            row = {}
            for name, value in zip(header[1:], columns[1:]):
                if value == "N/A":
                    continue
                row[name] = float(value)
            conditional_ler[bucket] = row

    return {
        "basis": basis,
        "total_samples": total_samples,
        "timing": timing,
        "overall_ler": overall_ler,
        "weight_counts": weight_counts,
        "conditional_ler": conditional_ler,
    }


def decoder_tail_us(parsed: dict[str, object], decoder_name: str) -> float:
    if decoder_name == "No-op":
        return 0.0
    key = DECODER_TO_TIMING_KEY.get(decoder_name)
    if key is None:
        return float("nan")
    timing = parsed["timing"]
    total_samples = parsed["total_samples"]
    if key not in timing or not total_samples:
        return float("nan")
    return float(timing[key]) / int(total_samples) * 1_000_000.0


def shared_tail_us(parsed: dict[str, object]) -> float:
    timing = parsed["timing"]
    total_samples = parsed["total_samples"]
    shared_keys = ("model_fwd", "residual_build")
    total = sum(float(timing.get(key, 0.0)) for key in shared_keys)
    return total / int(total_samples) * 1_000_000.0 if total_samples else float("nan")


def evaluate_policy(parsed: dict[str, object], fast_decoder: str, fallback_decoder: str, threshold: int) -> dict[str, float]:
    weight_counts = parsed["weight_counts"]
    conditional_ler = parsed["conditional_ler"]
    total_samples = float(parsed["total_samples"])
    if not weight_counts or not conditional_ler or total_samples <= 0:
        raise ValueError("ablation log is missing residual-weight statistics")

    fast_tail = decoder_tail_us(parsed, fast_decoder)
    fallback_tail = decoder_tail_us(parsed, fallback_decoder)
    shared_tail = shared_tail_us(parsed)
    weighted_ler = 0.0
    weighted_tail = 0.0
    routed_fast = 0.0

    for bucket, count in sorted(weight_counts.items()):
        prob = count / total_samples
        use_fast = bucket <= threshold
        chosen_decoder = fast_decoder if use_fast else fallback_decoder
        chosen_tail = fast_tail if use_fast else fallback_tail
        row = conditional_ler.get(bucket, {})
        if chosen_decoder not in row:
            chosen_ler = float(parsed["overall_ler"].get(chosen_decoder, math.nan))
        else:
            chosen_ler = float(row[chosen_decoder])
        weighted_ler += prob * chosen_ler
        weighted_tail += prob * chosen_tail
        if use_fast:
            routed_fast += prob

    return {
        "basis": str(parsed["basis"]),
        "fast_decoder": fast_decoder,
        "fallback_decoder": fallback_decoder,
        "threshold": threshold,
        "expected_ler": weighted_ler,
        "expected_tail_us": weighted_tail,
        "expected_total_us": shared_tail + weighted_tail,
        "shared_tail_us": shared_tail,
        "fraction_fast_path": routed_fast,
    }


def combine_by_policy(x_rows: list[dict[str, float]], z_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    z_index = {
        (row["fast_decoder"], row["fallback_decoder"], row["threshold"]): row
        for row in z_rows
    }
    combined = []
    for x_row in x_rows:
        key = (x_row["fast_decoder"], x_row["fallback_decoder"], x_row["threshold"])
        z_row = z_index.get(key)
        if not z_row:
            continue
        combined.append(
            {
                "fast_decoder": x_row["fast_decoder"],
                "fallback_decoder": x_row["fallback_decoder"],
                "threshold": x_row["threshold"],
                "x_expected_ler": x_row["expected_ler"],
                "z_expected_ler": z_row["expected_ler"],
                "avg_expected_ler": (x_row["expected_ler"] + z_row["expected_ler"]) / 2.0,
                "x_expected_total_us": x_row["expected_total_us"],
                "z_expected_total_us": z_row["expected_total_us"],
                "avg_expected_total_us": (x_row["expected_total_us"] + z_row["expected_total_us"]) / 2.0,
                "x_fraction_fast_path": x_row["fraction_fast_path"],
                "z_fraction_fast_path": z_row["fraction_fast_path"],
                "avg_fraction_fast_path": (x_row["fraction_fast_path"] + z_row["fraction_fast_path"]) / 2.0,
            }
        )
    combined.sort(key=lambda row: (row["avg_expected_ler"], row["avg_expected_total_us"]))
    return combined


def write_markdown(rows: list[dict[str, float]], path: Path) -> None:
    lines = [
        "| Fast decoder | Fallback decoder | Threshold | Avg expected LER | Avg expected total tail (µs) | Avg fast-path fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {fast} | {fallback} | {thr} | {ler:.6f} | {tail:.3f} | {frac:.3f} |".format(
                fast=row["fast_decoder"],
                fallback=row["fallback_decoder"],
                thr=int(row["threshold"]),
                ler=float(row["avg_expected_ler"]),
                tail=float(row["avg_expected_total_us"]),
                frac=float(row["avg_fraction_fast_path"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-log", required=True)
    parser.add_argument("--z-log", required=True)
    parser.add_argument("--fallback-decoder", default="Corr-PM")
    parser.add_argument("--fast-decoders", default="No-op,Union-Find,BP-only,BP+LSD-0,Uncorr-PM")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    x_parsed = parse_ablation_log(Path(args.x_log).resolve())
    z_parsed = parse_ablation_log(Path(args.z_log).resolve())
    fast_decoders = [item.strip() for item in args.fast_decoders.split(",") if item.strip()]
    thresholds = sorted(set(x_parsed["weight_counts"]) | set(z_parsed["weight_counts"]))

    x_rows = []
    z_rows = []
    for fast_decoder in fast_decoders:
        for threshold in thresholds:
            x_rows.append(evaluate_policy(x_parsed, fast_decoder, args.fallback_decoder, threshold))
            z_rows.append(evaluate_policy(z_parsed, fast_decoder, args.fallback_decoder, threshold))

    combined = combine_by_policy(x_rows, z_rows)

    output_json = Path(args.output_json).resolve() if args.output_json else Path(args.x_log).resolve().with_name("hybrid_policies.json")
    output_md = Path(args.output_md).resolve() if args.output_md else Path(args.x_log).resolve().with_name("hybrid_policies.md")
    payload = {
        "x_log": str(Path(args.x_log).resolve()),
        "z_log": str(Path(args.z_log).resolve()),
        "fallback_decoder": args.fallback_decoder,
        "policies": combined,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    write_markdown(combined[:15], output_md)
    print(f"[hybrid] wrote {output_json}")
    print(f"[hybrid] wrote {output_md}")
    if combined:
        best = combined[0]
        print(
            "[hybrid] best policy: "
            f"{best['fast_decoder']} <= w{int(best['threshold'])}, else {best['fallback_decoder']} | "
            f"avg_expected_ler={best['avg_expected_ler']:.6f} "
            f"avg_expected_total_us={best['avg_expected_total_us']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
