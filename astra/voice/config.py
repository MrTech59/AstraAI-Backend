from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
DEVICES_SCHEMA_PATH = CONFIG_DIR / "devices_schema.json"
DEVICES_SCHEMA_1_PATH = CONFIG_DIR / "devices_schema_1.json"


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    data: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value.strip())

    return data


def save_settings(settings: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in settings.items():
            handle.write(f"{key}: {_format_scalar(value)}\n")


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute():
        return candidate

    return ROOT_DIR / candidate


def to_relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()


def _parse_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).replace("\\", "/")
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'
