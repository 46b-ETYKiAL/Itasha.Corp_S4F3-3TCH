"""Prompt enhancement for ComfyUI workflows.

Adds model-aware quality tags, applies ComfyUI prompt weighting syntax,
handles BREAK-based conditioning regions, and recommends dynamic thresholding
for high CFG values.
"""

from __future__ import annotations

import re

from .presets import detect_model_family, get_preset

# ---------------------------------------------------------------------------
# Quality tag databases
# ---------------------------------------------------------------------------

_ANIME_QUALITY_TAGS = "masterpiece, best quality, highly detailed"
_PHOTO_QUALITY_TAGS = "professional photo, 8k uhd, sharp focus, high detail"
_SDXL_QUALITY_TAGS = "high quality, detailed, sharp"
_PONY_QUALITY_TAGS = "score_9, score_8_up, score_7_up"

_QUALITY_TAGS: dict[str, str] = {
    "sd15": _ANIME_QUALITY_TAGS,
    "sdxl": _SDXL_QUALITY_TAGS,
    "sdxl_turbo": _SDXL_QUALITY_TAGS,
    "pony": _PONY_QUALITY_TAGS,
    "sd3": "",
    "flux_dev": "",
    "flux_schnell": "",
    "flux2_dev": "",
    "flux2_klein": "",
    "cascade": _SDXL_QUALITY_TAGS,
}

_PHOTO_FAMILIES = {"sd15", "sdxl", "sdxl_turbo", "cascade"}

DYNAMIC_THRESHOLD_CFG = 10.0

# ---------------------------------------------------------------------------
# Weighting helpers
# ---------------------------------------------------------------------------


def apply_weight(token: str, weight: float = 1.5) -> str:
    """Apply ComfyUI prompt weighting syntax to a token.

    Uses the ``(token:weight)`` format recognized by ComfyUI.

    Args:
        token: Word or phrase to weight.
        weight: Emphasis weight (default 1.5).

    Returns:
        Weighted token string, e.g. ``"(sunset:1.5)"``.
    """
    if weight == 1.0:
        return token
    return f"({token}:{weight:.1f})"


def apply_break(segments: list[str]) -> str:
    """Join prompt segments with ComfyUI BREAK separator.

    BREAK creates separate conditioning regions, useful for compositional
    prompts with distinct subject areas.

    Args:
        segments: List of prompt text segments.

    Returns:
        Segments joined by `` BREAK ``.
    """
    return " BREAK ".join(s.strip() for s in segments if s.strip())


# ---------------------------------------------------------------------------
# Quality tag logic
# ---------------------------------------------------------------------------


def _get_quality_tags(family: str, style: str = "anime") -> str:
    """Get quality tags for a model family and style.

    For SD1.5 and compatible models, the style parameter selects between
    anime and photorealistic tag sets. Flux and SD3 return empty strings
    as they do not benefit from quality tags.

    Args:
        family: Model family string.
        style: Content style — ``"anime"`` or ``"photo"``.

    Returns:
        Comma-separated quality tag string, possibly empty.
    """
    if family in _QUALITY_TAGS:
        base = _QUALITY_TAGS[family]
        if family in _PHOTO_FAMILIES and style == "photo":
            return _PHOTO_QUALITY_TAGS
        return base
    return ""


# ---------------------------------------------------------------------------
# Dynamic thresholding
# ---------------------------------------------------------------------------


def recommend_dynamic_thresholding(cfg: float) -> bool:
    """Recommend whether dynamic thresholding should be enabled.

    High CFG values (>10) can cause color saturation and artifacts.
    Dynamic thresholding mitigates this.

    Args:
        cfg: CFG scale value.

    Returns:
        True if dynamic thresholding is recommended.
    """
    return cfg > DYNAMIC_THRESHOLD_CFG


# ---------------------------------------------------------------------------
# Core enhancement
# ---------------------------------------------------------------------------


def _clean_prompt(prompt: str) -> str:
    """Normalize whitespace and strip trailing commas.

    Args:
        prompt: Raw prompt text.

    Returns:
        Cleaned prompt string.
    """
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    return cleaned.rstrip(",").strip()


def enhance_prompt(
    prompt: str,
    model: str,
    *,
    style: str = "anime",
    add_quality_tags: bool = True,
    emphasis_words: dict[str, float] | None = None,
) -> str:
    """Enhance a prompt with model-aware quality tags and weighting.

    For Flux models (which use ``no_negative_prompt``), quality tags are
    not prepended since they don't improve output. Emphasis words are
    applied using ComfyUI ``(word:weight)`` syntax.

    Args:
        prompt: Original user prompt.
        model: Checkpoint filename for model family detection.
        style: Content style — ``"anime"`` or ``"photo"``.
        add_quality_tags: Whether to prepend quality tags.
        emphasis_words: Optional dict mapping words/phrases to emphasis
            weights, e.g. ``{"sunset": 1.5, "mountains": 1.2}``.

    Returns:
        Enhanced prompt string.
    """
    family = detect_model_family(model)
    preset = get_preset(family)
    cleaned = _clean_prompt(prompt)

    # Apply emphasis weighting to specific words
    if emphasis_words:
        for word, weight in emphasis_words.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            weighted = apply_weight(word, weight)
            cleaned = pattern.sub(weighted, cleaned)

    # Skip quality tags for models that don't benefit
    if preset.no_negative_prompt or not add_quality_tags:
        return cleaned

    quality = _get_quality_tags(family, style)
    if quality:
        return f"{quality}, {cleaned}"
    return cleaned


def build_negative_prompt(
    model: str,
    *,
    extra_negatives: str = "",
) -> str:
    """Build a model-aware negative prompt.

    For Flux and SD3, returns empty string since these architectures do
    not use or benefit from negative prompts.

    Args:
        model: Checkpoint filename for model family detection.
        extra_negatives: Additional negative terms to append.

    Returns:
        Negative prompt string, possibly empty.
    """
    family = detect_model_family(model)
    preset = get_preset(family)

    if preset.no_negative_prompt:
        return ""

    base = preset.negative_template
    if extra_negatives:
        if base:
            return f"{base}, {extra_negatives.strip()}"
        return extra_negatives.strip()
    return base
