from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import RegimeSpec

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() in {"null", "none"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported line in simple YAML parser: {raw_line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)
    return root


def repo_root_from_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is not None:
        payload = yaml.safe_load(path.read_text()) or {}
    else:
        payload = _simple_yaml_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def default_catalog_path(repo_root: Path | None = None) -> Path:
    return repo_root_from_path(repo_root) / "conf" / "regime_catalog.yaml"


def default_router_config_path(repo_root: Path | None = None) -> Path:
    return repo_root_from_path(repo_root) / "conf" / "regime_router.yaml"


def load_regime_catalog(path: Path) -> dict[str, RegimeSpec]:
    payload = load_yaml(path)
    regimes = payload.get("regimes", {})
    out: dict[str, RegimeSpec] = {}
    for name, row in regimes.items():
        if not isinstance(row, dict):
            raise ValueError(f"regime {name} must be a mapping")
        out[name] = RegimeSpec(
            name=name,
            config_name=str(row["config_name"]),
            experiment_name=str(row["experiment_name"]),
            description=str(row.get("description", "")),
            default=bool(row.get("default", False)),
            artifacts=dict(row.get("artifacts", {})),
            metadata=dict(row.get("metadata", {})),
        )
    if not out:
        raise ValueError(f"no regimes found in {path}")
    return out


def load_router_config(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def choose_default_regime(
    catalog: dict[str, RegimeSpec],
    router_cfg: dict[str, Any] | None = None,
) -> str:
    if router_cfg:
        configured = str(router_cfg.get("default_regime", "")).strip()
        if configured:
            if configured not in catalog:
                raise KeyError(f"default_regime {configured!r} not present in catalog")
            return configured
    for name, spec in catalog.items():
        if spec.default:
            return name
    raise ValueError("no default regime configured")
