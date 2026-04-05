"""Upscaling workflow builder for ComfyUI.

Provides automatic upscaler selection based on content type, tiled diffusion
for high-resolution sources on limited VRAM, and ComfyUI API-format workflow
generation for 2x and 4x upscaling.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANIME_UPSCALERS = ("4x-UltraSharp", "RealESRGAN_x4plus_anime_6B")
PHOTO_UPSCALERS = ("SwinIR_4x", "4x-UltraSharp")

VALID_SCALE_FACTORS = (2, 4)

TILE_THRESHOLD_PX = 1024
VRAM_TILE_THRESHOLD_GB = 12

DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64

# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------


def _is_anime_content(content_type: str) -> bool:
    """Check whether the content type indicates anime/illustration style.

    Args:
        content_type: Descriptive string such as ``"anime"``,
            ``"illustration"``, ``"photorealistic"``, or ``"photo"``.

    Returns:
        True if the content is anime/illustration style.
    """
    return content_type.lower() in {"anime", "illustration", "cartoon", "manga"}


def select_upscaler(content_type: str, scale: int = 4) -> str:
    """Auto-select the best upscaler model for the given content type.

    Args:
        content_type: Content style — ``"anime"``, ``"photo"``, etc.
        scale: Target scale factor (2 or 4).

    Returns:
        Upscaler model name string.

    Raises:
        ValueError: If ``scale`` is not 2 or 4.
    """
    if scale not in VALID_SCALE_FACTORS:
        raise ValueError(
            f"Unsupported scale factor {scale}. Must be one of {VALID_SCALE_FACTORS}."
        )

    if _is_anime_content(content_type):
        return ANIME_UPSCALERS[0]
    return PHOTO_UPSCALERS[0]


# ---------------------------------------------------------------------------
# Tiled mode detection
# ---------------------------------------------------------------------------


def needs_tiled_mode(
    source_width: int,
    source_height: int,
    vram_gb: float = 8.0,
) -> bool:
    """Determine whether tiled diffusion should be used.

    Tiled mode is recommended when the source image exceeds the tile
    threshold on either axis AND available VRAM is below the VRAM
    threshold.

    Args:
        source_width: Source image width in pixels.
        source_height: Source image height in pixels.
        vram_gb: Available GPU VRAM in gigabytes.

    Returns:
        True if tiled mode is recommended.
    """
    large_source = source_width > TILE_THRESHOLD_PX or source_height > TILE_THRESHOLD_PX
    low_vram = vram_gb < VRAM_TILE_THRESHOLD_GB
    return large_source and low_vram


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _node(class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Build a single ComfyUI node dict.

    Args:
        class_type: ComfyUI node class name.
        inputs: Node input parameters.

    Returns:
        Node dict.
    """
    return {"class_type": class_type, "inputs": inputs}


def _ref(node_id: str, output_index: int = 0) -> list[str | int]:
    """Create a ComfyUI node output reference.

    Args:
        node_id: Target node ID.
        output_index: Output slot index.

    Returns:
        Reference list ``[node_id, output_index]``.
    """
    return [node_id, output_index]


# ---------------------------------------------------------------------------
# Workflow builders
# ---------------------------------------------------------------------------


def _build_simple_upscale(
    image_path: str,
    upscaler_model: str,
) -> dict[str, dict[str, Any]]:
    """Build a simple model-based upscale workflow.

    Args:
        image_path: Path to the source image.
        upscaler_model: Upscaler model name.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("LoadImage", {"image": image_path})
    nodes["2"] = _node("UpscaleModelLoader", {"model_name": upscaler_model})
    nodes["3"] = _node(
        "ImageUpscaleWithModel",
        {
            "upscale_model": _ref("2", 0),
            "image": _ref("1", 0),
        },
    )
    nodes["4"] = _node(
        "SaveImage",
        {
            "images": _ref("3", 0),
            "filename_prefix": "s4f3_upscaled",
        },
    )
    return nodes


def _build_tiled_upscale(
    image_path: str,
    upscaler_model: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> dict[str, dict[str, Any]]:
    """Build a tiled upscale workflow for large images on low VRAM.

    Uses ``ImageScaleBy`` with tiling to avoid OOM on large sources.

    Args:
        image_path: Path to the source image.
        upscaler_model: Upscaler model name.
        tile_size: Tile width/height in pixels.
        tile_overlap: Overlap between tiles in pixels.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("LoadImage", {"image": image_path})
    nodes["2"] = _node("UpscaleModelLoader", {"model_name": upscaler_model})
    nodes["3"] = _node(
        "UltimateSDUpscale",
        {
            "upscale_model": _ref("2", 0),
            "image": _ref("1", 0),
            "tile_width": tile_size,
            "tile_height": tile_size,
            "tile_overlap": tile_overlap,
            "seam_fix_mode": "Band Pass",
            "seam_fix_width": 64,
            "seam_fix_denoise": 0.35,
        },
    )
    nodes["4"] = _node(
        "SaveImage",
        {
            "images": _ref("3", 0),
            "filename_prefix": "s4f3_upscaled_tiled",
        },
    )
    return nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_upscale_workflow(
    image_path: str,
    scale: int = 4,
    content_type: str = "photo",
    *,
    source_width: int = 512,
    source_height: int = 512,
    vram_gb: float = 8.0,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> dict[str, Any]:
    """Build a complete upscale workflow in ComfyUI API format.

    Automatically selects the upscaler model and decides whether to use
    tiled processing based on source dimensions and VRAM.

    Args:
        image_path: Path to the source image file.
        scale: Scale factor (2 or 4).
        content_type: Content style for upscaler selection.
        source_width: Source image width in pixels.
        source_height: Source image height in pixels.
        vram_gb: Available GPU VRAM in gigabytes.
        tile_size: Tile size for tiled mode.
        tile_overlap: Tile overlap for tiled mode.

    Returns:
        ComfyUI API-format workflow dict.

    Raises:
        ValueError: If ``scale`` is invalid.
    """
    upscaler = select_upscaler(content_type, scale)

    if needs_tiled_mode(source_width, source_height, vram_gb):
        nodes = _build_tiled_upscale(image_path, upscaler, tile_size, tile_overlap)
    else:
        nodes = _build_simple_upscale(image_path, upscaler)

    return {
        "prompt": nodes,
        "metadata": {
            "upscaler": upscaler,
            "scale": scale,
            "content_type": content_type,
            "tiled": needs_tiled_mode(source_width, source_height, vram_gb),
        },
    }
