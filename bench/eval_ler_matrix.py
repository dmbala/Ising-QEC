#!/usr/bin/env python3
"""Run or aggregate train-vs-test LER matrix evaluations for Ising-QEC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
DEFAULT_CONFIGS = (
    "config_ising_qec_d5_biased",
    "config_ising_qec_d5_bias3",
    "config_ising_qec_d5_bias30",
    "config_ising_qec_d5_drift",
    "config_ising_qec_d5_hotspot",
)
CSV_FIELDS = [
    "train_experiment",
    "eval_config",
    "status",
    "returncode",
    "wall_seconds",
    "samples",
    "onnx_workflow",
    "quant_format",
    "x_ler_baseline",
    "x_ler_after",
    "z_ler_baseline",
    "z_ler_after",
    "avg_ler_baseline",
    "avg_ler_after",
    "x_latency_baseline_us",
    "x_latency_after_us",
    "z_latency_baseline_us",
    "z_latency_after_us",
    "avg_latency_baseline_us",
    "avg_latency_after_us",
    "avg_speedup",
    "raw_log",
    "summary_json",
]


@dataclass
class MatrixCase:
    train_experiment: str
    eval_config: str


def config_to_experiment(config_name: str) -> str:
    suffix = config_name
    prefix = "config_ising_qec_d5_"
    if suffix.startswith(prefix):
        suffix = suffix[len(prefix):]
    return f"qec-decoder-d5-{suffix.replace('_', '-')}"


def parse_csv_list(raw: str | None, default: Iterable[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_config_symlinks(repo_root: Path, ising_dir: Path, config_names: Iterable[str]) -> None:
    src_dir = repo_root / "conf"
    dst_dir = ising_dir / "conf"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for config_name in config_names:
        src = src_dir / f"{config_name}.yaml"
        if not src.exists():
            raise FileNotFoundError(f"missing local config: {src}")
        dst = dst_dir / src.name
        if dst.is_symlink() and dst.resolve() == src.resolve():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def read_manifest(path: Path) -> list[MatrixCase]:
    payload = json.loads(path.read_text())
    cases = []
    for row in payload:
        cases.append(
            MatrixCase(
                train_experiment=row["train_experiment"],
                eval_config=row["eval_config"],
            )
        )
    return cases


def build_cases(train_experiments: list[str], eval_configs: list[str]) -> list[MatrixCase]:
    return [
        MatrixCase(train_experiment=train_experiment, eval_config=eval_config)
        for train_experiment in train_experiments
        for eval_config in eval_configs
    ]


def parse_metrics(stdout: str) -> dict[str, float]:
    patterns = {
        "x_latency": rf"PyMatching latency - X basis \(µs/round\):\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "z_latency": rf"PyMatching latency - Z basis \(µs/round\):\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "avg_latency": rf"PyMatching latency - Avg \(µs/round\):\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "x_ler": rf"LER - X basis:\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "z_ler": rf"LER - Z basis:\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "avg_ler": rf"LER - Avg:\s+{FLOAT_RE}\s+{FLOAT_RE}",
        "speedup": rf"PyMatching speedup \(Avg X/Z\):\s+{FLOAT_RE}x",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"could not parse {key} from inference output")
        if key == "speedup":
            out["avg_speedup"] = float(match.group(1))
            continue
        left = float(match.group(1))
        right = float(match.group(2))
        if key.endswith("_latency"):
            stem = key[:-8]
            out[f"{stem}_latency_baseline_us"] = left
            out[f"{stem}_latency_after_us"] = right
        else:
            stem = key[:-4]
            out[f"{stem}_ler_baseline"] = left
            out[f"{stem}_ler_after"] = right
    return out


def run_case(
    case: MatrixCase,
    *,
    repo_root: Path,
    ising_dir: Path,
    sif: Path,
    shared_output_dir: Path,
    results_root: Path,
    samples: int,
    latency_samples: int,
    onnx_workflow: int,
    quant_format: str,
    dry_run: bool,
) -> dict[str, object]:
    raw_dir = results_root / "raw"
    json_dir = results_root / "json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{case.train_experiment}__{case.eval_config}"
    log_path = raw_dir / f"{tag}.log"
    summary_path = json_dir / f"{tag}.json"

    cmd = [
        "singularity",
        "exec",
        "--nv",
        "--bind",
        f"{ising_dir}:{ising_dir}",
        "--bind",
        f"{shared_output_dir}:{shared_output_dir}",
        "--env",
        f"SHARED_OUTPUT_DIR={shared_output_dir}",
        "--env",
        f"EXPERIMENT_NAME={case.train_experiment}",
        "--env",
        f"CONFIG_NAME={case.eval_config}",
        "--env",
        "WORKFLOW=inference",
        "--env",
        "GPUS=1",
        "--env",
        "FRESH_START=0",
        "--env",
        f"ONNX_WORKFLOW={onnx_workflow}",
        "--env",
        f"PREDECODER_INFERENCE_NUM_SAMPLES={samples}",
        "--env",
        f"PREDECODER_INFERENCE_LATENCY_SAMPLES={latency_samples}",
        "--env",
        "PREDECODER_INFERENCE_MEAS_BASIS=both",
        "--env",
        "PREDECODER_INFERENCE_NUM_WORKERS=0",
        "--env",
        "PREDECODER_DISABLE_SDR=1",
        "--env",
        "PREDECODER_TORCH_COMPILE=0",
        "--env",
        "PREDECODER_VERBOSE=1",
    ]
    if quant_format:
        cmd.extend(["--env", f"QUANT_FORMAT={quant_format}"])
    cmd.extend([str(sif), "bash", str(ising_dir / "code" / "scripts" / "cluster_train.sh")])

    record: dict[str, object] = {
        "train_experiment": case.train_experiment,
        "eval_config": case.eval_config,
        "samples": samples,
        "onnx_workflow": onnx_workflow,
        "quant_format": quant_format or "",
        "raw_log": str(log_path),
        "summary_json": str(summary_path),
    }
    if dry_run:
        record["status"] = "dry_run"
        record["command"] = cmd
        summary_path.write_text(json.dumps(record, indent=2))
        return record

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    wall_seconds = time.perf_counter() - t0
    combined_output = proc.stdout
    if proc.stderr:
        combined_output += "\n[stderr]\n" + proc.stderr
    log_path.write_text(combined_output)

    record["returncode"] = proc.returncode
    record["wall_seconds"] = round(wall_seconds, 3)

    if proc.returncode != 0:
        record["status"] = "command_failed"
        summary_path.write_text(json.dumps(record, indent=2))
        return record

    try:
        record.update(parse_metrics(proc.stdout))
        record["status"] = "ok"
    except ValueError as exc:
        record["status"] = "parse_failed"
        record["parse_error"] = str(exc)

    summary_path.write_text(json.dumps(record, indent=2))
    return record


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC")
    parser.add_argument("--ising-dir", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC/Ising-Decoding")
    parser.add_argument("--sif", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC/env/ising.sif")
    parser.add_argument("--shared-output-dir", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC/outputs")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--train-experiments", default="")
    parser.add_argument("--eval-configs", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--latency-samples", type=int, default=0)
    parser.add_argument("--onnx-workflow", type=int, default=0)
    parser.add_argument("--quant-format", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ising_dir = Path(args.ising_dir).resolve()
    sif = Path(args.sif).resolve()
    shared_output_dir = Path(args.shared_output_dir).resolve()

    if args.results_root:
        results_root = Path(args.results_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = repo_root / "results" / "matrix" / stamp
    results_root.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        cases = read_manifest(Path(args.manifest).resolve())
    else:
        eval_configs = parse_csv_list(args.eval_configs, DEFAULT_CONFIGS)
        train_experiments = parse_csv_list(
            args.train_experiments,
            [config_to_experiment(config_name) for config_name in DEFAULT_CONFIGS],
        )
        cases = build_cases(train_experiments, eval_configs)

    manifest_path = results_root / "manifest.json"
    manifest_path.write_text(json.dumps([asdict(case) for case in cases], indent=2))

    ensure_config_symlinks(repo_root, ising_dir, {case.eval_config for case in cases})

    rows = []
    for case in cases:
        row = run_case(
            case,
            repo_root=repo_root,
            ising_dir=ising_dir,
            sif=sif,
            shared_output_dir=shared_output_dir,
            results_root=results_root,
            samples=args.samples,
            latency_samples=args.latency_samples,
            onnx_workflow=args.onnx_workflow,
            quant_format=args.quant_format,
            dry_run=args.dry_run,
        )
        rows.append(row)
        status = row["status"]
        print(f"[matrix] {case.train_experiment} vs {case.eval_config}: {status}")

    write_csv(rows, results_root / "summary.csv")
    (results_root / "summary.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else "")
    )
    print(f"[matrix] wrote {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
