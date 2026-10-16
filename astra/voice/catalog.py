from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ROOT_DIR, resolve_project_path, to_relative_project_path

MODELS_DIR = ROOT_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
VOICES_DIR = MODELS_DIR / "voices"


@dataclass(frozen=True)
class AssetEntry:
    path: Path
    relative_path: str
    size_bytes: int
    metadata_path: Path | None = None

    @property
    def name(self) -> str:
        return self.path.stem


def discover_voice_models() -> list[AssetEntry]:
    return _discover_assets(VOICES_DIR, "*.onnx", include_metadata=True)


def discover_llm_models() -> list[AssetEntry]:
    return _discover_assets(LLM_DIR, "*.gguf", include_metadata=False)


def _discover_assets(directory: Path, pattern: str, include_metadata: bool) -> list[AssetEntry]:
    if not directory.exists():
        return []

    entries: list[AssetEntry] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue

        metadata_path = Path(f"{path}.json") if include_metadata else None
        entries.append(
            AssetEntry(
                path=path,
                relative_path=to_relative_project_path(path),
                size_bytes=path.stat().st_size,
                metadata_path=metadata_path,
            )
        )

    return entries


def find_current_entry(entries: Iterable[AssetEntry], configured_path: Any) -> AssetEntry | None:
    resolved = resolve_project_path(configured_path)
    if resolved is None:
        return None

    for entry in entries:
        if entry.path.resolve() == resolved.resolve():
            return entry

    return None


def select_entry(entries: list[AssetEntry], selection: str) -> AssetEntry | None:
    token = selection.strip()
    if not token:
        return None

    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(entries):
            return entries[index]

    token = token.lower().replace("\\", "/")
    for entry in entries:
        candidates = {
            entry.name.lower(),
            entry.path.name.lower(),
            entry.relative_path.lower(),
        }
        if token in candidates:
            return entry

    return None


def format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size_bytes} B"


def sample_is_all_zero(path: Path, sample_size: int = 4096) -> bool:
    if not path.exists() or not path.is_file():
        return False

    with path.open("rb") as handle:
        chunk = handle.read(sample_size)

    return bool(chunk) and all(byte == 0 for byte in chunk)


def validate_settings_assets(settings: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    # Local LLM file only applies to the "local" backend; NVIDIA NIM is remote.
    if settings.get("llm_type", "local") == "local":
        llm_path = resolve_project_path(settings.get("llm_model_path"))
        _validate_asset("Configured llm model", llm_path, issues)
    else:
        for key, label in (("nvidia_model", "Configured llm model"), ("nvidia_api_key", "Nvidia API key")):
            if not settings.get(key):
                issues.append(f"{label} is not set")

    voice_path = resolve_project_path(settings.get("tts_voice_path"))
    _validate_asset("Configured voice model", voice_path, issues)

    if voice_path is not None:
        metadata_path = Path(f"{voice_path}.json")
        if not metadata_path.exists():
            issues.append(
                f"voice metadata file is missing: {metadata_path.relative_to(ROOT_DIR).as_posix()}"
            )
        elif sample_is_all_zero(metadata_path):
            issues.append(
                "voice metadata file looks zero-filled: "
                f"{metadata_path.relative_to(ROOT_DIR).as_posix()}"
            )

    return issues


def _validate_asset(label: str, path: Path | None, issues: list[str]) -> None:
    if path is None:
        issues.append(f"{label} is not set")
        return

    if not path.exists():
        issues.append(f"{label} is missing: {path.relative_to(ROOT_DIR).as_posix()}")
        return

    if path.stat().st_size == 0:
        issues.append(f"{label} is empty: {path.relative_to(ROOT_DIR).as_posix()}")
