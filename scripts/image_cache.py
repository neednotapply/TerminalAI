"""Shared helpers for caching generated images and metadata."""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


def ensure_images_dir(base_data: Path) -> Path:
    """Return the directory used to cache generated images."""

    images_dir = Path(base_data) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def resolve_cached_paths(
    images_dir: Path, image_name: str, *, default_suffix: str = ".png"
) -> Tuple[Path, Path]:
    """Return the cache paths for an image and its metadata."""

    raw_text = image_name if isinstance(image_name, str) else ""
    base_name = Path(raw_text).name or raw_text
    base_path = Path(base_name)
    stem = base_path.stem or base_name or "image"
    sanitized_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._")
    if not sanitized_stem:
        sanitized_stem = hashlib.sha256(raw_text.encode("utf-8", "ignore")).hexdigest()[
            :16
        ]

    suffix_candidate = base_path.suffix
    if suffix_candidate and re.fullmatch(r"\.[0-9A-Za-z]+", suffix_candidate):
        suffix = suffix_candidate
    else:
        suffix = default_suffix

    image_path = images_dir / f"{sanitized_stem}{suffix}"
    metadata_path = image_path.with_suffix(".json")
    return image_path, metadata_path


def normalize_metadata_value(value: Any) -> Any:
    """Convert metadata values into JSON serialisable primitives."""

    if isinstance(value, dict):
        return {key: normalize_metadata_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_metadata_value(item) for item in value]
    if isinstance(value, datetime.datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
        return dt.isoformat()
    return value


def write_metadata_file(path: Path, metadata: Dict[str, Any]) -> None:
    """Persist image metadata alongside the cached image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
