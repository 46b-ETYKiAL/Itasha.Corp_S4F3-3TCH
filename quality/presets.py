"""Model-specific quality preset library for ComfyUI workflows.

Loads quality presets from YAML configuration and provides model family
detection from checkpoint filenames. Each preset contains optimal sampler,
scheduler, CFG, and step settings for a given model architecture.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRESETS_YAML = (
    Path(__file__).resolve().parents[3]
    / "Libraries"
    / "workflows"
    / "ComfyUI"
    / "config"
    / "quality-presets.yaml"
)

# Ordered list — more specific patterns first to avoid false positives.
_MODEL_PATTERNS: list[tuple[str, str]] = [
    (r"flux[\-_]?2[\-_]?klein|flux2[\-_]?klein", "flux2_klein"),
    (r"flux[\-_]?2[\-_]?dev|flux2[\-_]?dev", "flux2_dev"),
    (r"schnell", "flux_schnell"),
    (r"flux[\-_]?dev|flux(?!.*schnell)", "flux_dev"),
    (r"sdxl[\-_]?turbo|turbo", "sdxl_turbo"),
    (r"cascade", "cascade"),
    (r"pony", "pony"),
    (r"sd[\-_]?3|sd3", "sd3"),
    (r"sdxl|sd[\-_]?xl|xl", "sdxl"),
]

_DEFAULT_FAMILY = "sd15"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QualityPreset:
    """Optimal generation parameters for a model architecture."""

    sampler: str
    scheduler: str
    cfg: float
    steps: int
    cfg_range: tuple[float, float]
    steps_range: tuple[int, int]
    vae: str = ""
    negative_template: str = ""
    uses_t5: bool = False
    no_negative_prompt: bool = False
    triple_clip: bool = False
    multi_reference: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _midpoint_int(low: int, high: int) -> int:
    """Return the midpoint of two integers, rounded down."""
    return (low + high) // 2


def _midpoint_float(low: float, high: float) -> float:
    """Return the midpoint of two floats, rounded to one decimal."""
    return round((low + high) / 2, 1)


def _load_presets_yaml(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the quality-presets YAML file.

    Args:
        path: Override path for testing. Defaults to the canonical location.

    Returns:
        Parsed YAML dict with a ``presets`` key.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    yaml_path = path or _PRESETS_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"Quality presets YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "presets" not in data:
        raise ValueError(
            f"Invalid quality-presets YAML: missing top-level 'presets' key in {yaml_path}"
        )
    return data


def _build_preset(name: str, raw: dict[str, Any]) -> QualityPreset:
    """Convert a raw YAML preset dict into a ``QualityPreset``.

    Args:
        name: Preset name (used in error messages).
        raw: Raw dict from YAML.

    Returns:
        Populated ``QualityPreset`` instance.

    Raises:
        ValueError: If required fields are missing.
    """
    required = ("sampler", "scheduler", "cfg_range", "steps_range")
    missing = [f for f in required if f not in raw]
    if missing:
        raise ValueError(f"Preset '{name}' missing required fields: {missing}")

    cfg_range = (float(raw["cfg_range"][0]), float(raw["cfg_range"][1]))
    steps_range = (int(raw["steps_range"][0]), int(raw["steps_range"][1]))

    return QualityPreset(
        sampler=raw["sampler"],
        scheduler=raw["scheduler"],
        cfg=_midpoint_float(*cfg_range),
        steps=_midpoint_int(*steps_range),
        cfg_range=cfg_range,
        steps_range=steps_range,
        vae=raw.get("vae", ""),
        negative_template=raw.get("negative_template", ""),
        uses_t5=bool(raw.get("uses_t5", False)),
        no_negative_prompt=bool(raw.get("no_negative_prompt", False)),
        triple_clip=bool(raw.get("triple_clip", False)),
        multi_reference=bool(raw.get("multi_reference", False)),
    )


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_preset_cache: dict[str, QualityPreset] | None = None


def _get_all_presets(yaml_path: Path | None = None) -> dict[str, QualityPreset]:
    """Return all presets, loading from YAML on first call.

    Args:
        yaml_path: Override path for testing.

    Returns:
        Dict mapping model family name to ``QualityPreset``.
    """
    global _preset_cache
    if _preset_cache is not None and yaml_path is None:
        return _preset_cache

    data = _load_presets_yaml(yaml_path)
    presets = {name: _build_preset(name, raw) for name, raw in data["presets"].items()}

    if yaml_path is None:
        _preset_cache = presets
    return presets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_model_family(checkpoint_name: str) -> str:
    """Detect the model family from a checkpoint filename.

    Uses ordered regex matching against known patterns. Falls back to
    ``sd15`` when no pattern matches.

    Args:
        checkpoint_name: Filename (with or without extension) of the
            checkpoint, e.g. ``"dreamshaper_8_sdxl.safetensors"``.

    Returns:
        Model family string such as ``"sdxl"``, ``"flux_dev"``, ``"sd15"``.
    """
    lower = checkpoint_name.lower()
    for pattern, family in _MODEL_PATTERNS:
        if re.search(pattern, lower):
            return family
    return _DEFAULT_FAMILY


def get_preset(model_family: str, *, yaml_path: Path | None = None) -> QualityPreset:
    """Return the quality preset for a given model family.

    Args:
        model_family: One of the keys in ``quality-presets.yaml``
            (e.g. ``"sdxl"``, ``"flux_dev"``).
        yaml_path: Override YAML path for testing.

    Returns:
        ``QualityPreset`` for the requested family.

    Raises:
        KeyError: If ``model_family`` is not found in presets.
    """
    presets = _get_all_presets(yaml_path)
    if model_family not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown model family '{model_family}'. Available: {available}")
    return presets[model_family]


def get_preset_for_checkpoint(
    checkpoint_name: str, *, yaml_path: Path | None = None
) -> QualityPreset:
    """Detect model family from checkpoint name and return its preset.

    Convenience wrapper combining ``detect_model_family`` and ``get_preset``.

    Args:
        checkpoint_name: Checkpoint filename.
        yaml_path: Override YAML path for testing.

    Returns:
        ``QualityPreset`` for the detected model family.
    """
    family = detect_model_family(checkpoint_name)
    return get_preset(family, yaml_path=yaml_path)


def list_families(*, yaml_path: Path | None = None) -> list[str]:
    """Return sorted list of available model family names.

    Args:
        yaml_path: Override YAML path for testing.

    Returns:
        Sorted list of family name strings.
    """
    return sorted(_get_all_presets(yaml_path))


def clear_cache() -> None:
    """Clear the in-memory preset cache (useful for testing)."""
    global _preset_cache
    _preset_cache = None
