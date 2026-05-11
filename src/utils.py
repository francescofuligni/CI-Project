from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            data = yaml.safe_load(handle)
            return data or {}
        return _load_simple_yaml(handle.readlines())


def _load_simple_yaml(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value:
            data[key.strip()] = value
    return data


def resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()
