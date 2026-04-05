"""ComfyUI output quality optimization package.

Provides model-aware quality presets, workflow generation, prompt
enhancement, and upscaling for ComfyUI image generation pipelines.

Public API:
    - ``get_optimal_settings(model)`` — Returns quality preset for a model.
    - ``enhance_prompt(prompt, model)`` — Enhance prompt with quality tags.
    - ``build_upscale_workflow(image_path, scale, content_type)`` — Upscale workflow.
    - ``generate_optimized(prompt, model)`` — Full optimized txt2img workflow.
"""

from __future__ import annotations

from typing import Any

from .presets import QualityPreset, detect_model_family, get_preset_for_checkpoint
from .prompt_enhancer import enhance_prompt as _enhance_prompt
from .upscaler import build_upscale_workflow as _build_upscale_workflow
from .workflow_builder import WorkflowBuilder

__all__ = [
    "QualityPreset",
    "build_upscale_workflow",
    "detect_model_family",
    "enhance_prompt",
    "generate_optimized",
    "get_optimal_settings",
]

# Singleton workflow builder instance.
_builder = WorkflowBuilder()


def get_optimal_settings(model: str) -> QualityPreset:
    """Return the optimal quality preset for a model checkpoint.

    Args:
        model: Checkpoint filename (e.g. ``"dreamshaper_8.safetensors"``).

    Returns:
        ``QualityPreset`` with sampler, scheduler, CFG, steps, etc.
    """
    return get_preset_for_checkpoint(model)


def enhance_prompt(prompt: str, model: str, **kwargs: Any) -> str:
    """Enhance a prompt with model-aware quality tags and weighting.

    Args:
        prompt: Original user prompt.
        model: Checkpoint filename.
        **kwargs: Passed through to ``prompt_enhancer.enhance_prompt``.

    Returns:
        Enhanced prompt string.
    """
    return _enhance_prompt(prompt, model, **kwargs)


def build_upscale_workflow(
    image_path: str,
    scale: int = 4,
    content_type: str = "photo",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an upscale workflow in ComfyUI API format.

    Args:
        image_path: Path to the source image.
        scale: Scale factor (2 or 4).
        content_type: Content style for upscaler selection.
        **kwargs: Passed through to ``upscaler.build_upscale_workflow``.

    Returns:
        ComfyUI API-format workflow dict.
    """
    return _build_upscale_workflow(image_path, scale, content_type, **kwargs)


def generate_optimized(
    prompt: str,
    model: str,
    *,
    width: int = 1024,
    height: int = 1024,
    style: str = "anime",
    seed: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    """Generate a fully optimized txt2img workflow.

    Combines prompt enhancement and workflow building with model-aware
    quality presets applied automatically.

    Args:
        prompt: Original user prompt.
        model: Checkpoint filename.
        width: Output image width.
        height: Output image height.
        style: Content style for quality tag selection.
        seed: Random seed.
        **overrides: Override preset values.

    Returns:
        ComfyUI API-format workflow dict with enhanced prompt.
    """
    enhanced = _enhance_prompt(prompt, model, style=style)
    return _builder.build_txt2img(
        enhanced, model, width, height, seed=seed, **overrides
    )
