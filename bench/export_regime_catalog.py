#!/usr/bin/env python3
"""Export the regime catalog with resolved artifact metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_router.catalog import default_catalog_path, load_regime_catalog


def _render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Regime Catalog",
        "",
        "| Regime | Config | Experiment | Checkpoint glob | Engine template |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {cfg} | {exp} | `{ckpt}` | `{engine}` |".format(
                name=row["name"],
                cfg=row["config_name"],
                exp=row["experiment_name"],
                ckpt=row["checkpoint_glob"],
                engine=row["engine_template"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--catalog", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    catalog_path = Path(args.catalog).resolve() if args.catalog else default_catalog_path(repo_root)
    catalog = load_regime_catalog(catalog_path)
    rows = []
    for name, spec in sorted(catalog.items()):
        rows.append(
            {
                "name": name,
                "config_name": spec.config_name,
                "experiment_name": spec.experiment_name,
                "description": spec.description,
                "checkpoint_glob": spec.artifacts.get("checkpoint_glob", ""),
                "engine_template": spec.artifacts.get("engine_template", ""),
                "default": spec.default,
            }
        )

    json_out = Path(args.json_out).resolve() if args.json_out else repo_root / "results" / "router" / "catalog" / "regime_catalog.json"
    md_out = Path(args.md_out).resolve() if args.md_out else repo_root / "results" / "router" / "catalog" / "regime_catalog.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2))
    md_out.write_text(_render_markdown(rows))
    print(f"[catalog] wrote {json_out}")
    print(f"[catalog] wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
