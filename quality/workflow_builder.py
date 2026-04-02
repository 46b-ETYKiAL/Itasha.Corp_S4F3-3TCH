"""Model-aware ComfyUI workflow builder.

Generates ComfyUI API-format workflow dicts for txt2img and img2img
operations across SD1.5, SDXL, Flux, SD3, Pony, and Cascade architectures.
Uses quality presets as defaults with override support.
"""

from __future__ import annotations

from typing import Any

from .presets import QualityPreset, detect_model_family, get_preset

# ---------------------------------------------------------------------------
# Internal node helpers
# ---------------------------------------------------------------------------


def _node(class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Build a single ComfyUI node dict.

    Args:
        class_type: ComfyUI node class name.
        inputs: Node input parameters.

    Returns:
        Node dict with ``class_type`` and ``inputs`` keys.
    """
    return {"class_type": class_type, "inputs": inputs}


def _ref(node_id: str, output_index: int = 0) -> list[str | int]:
    """Create a ComfyUI node reference (edge).

    Args:
        node_id: Target node ID string.
        output_index: Output slot index on the target node.

    Returns:
        Two-element list ``[node_id, output_index]``.
    """
    return [node_id, output_index]


# ---------------------------------------------------------------------------
# Architecture-specific builders
# ---------------------------------------------------------------------------


def _build_sd15_nodes(
    prompt: str,
    preset: QualityPreset,
    width: int,
    height: int,
    model_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build SD1.5 txt2img nodes: checkpoint, CLIP encode, KSampler, VAE decode.

    Args:
        prompt: Positive prompt text.
        preset: Quality preset with sampler/scheduler/cfg/steps.
        width: Output image width.
        height: Output image height.
        model_name: Checkpoint filename.
        seed: Random seed.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("CheckpointLoaderSimple", {"ckpt_name": model_name})
    nodes["2"] = _node(
        "CLIPTextEncode",
        {
            "text": prompt,
            "clip": _ref("1", 1),
        },
    )
    nodes["3"] = _node(
        "CLIPTextEncode",
        {
            "text": preset.negative_template,
            "clip": _ref("1", 1),
        },
    )
    nodes["4"] = _node(
        "EmptyLatentImage",
        {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    )
    nodes["5"] = _node(
        "KSampler",
        {
            "model": _ref("1", 0),
            "positive": _ref("2", 0),
            "negative": _ref("3", 0),
            "latent_image": _ref("4", 0),
            "seed": seed,
            "steps": preset.steps,
            "cfg": preset.cfg,
            "sampler_name": preset.sampler,
            "scheduler": preset.scheduler,
            "denoise": 1.0,
        },
    )
    nodes["6"] = _node(
        "VAEDecode",
        {
            "samples": _ref("5", 0),
            "vae": _ref("1", 2),
        },
    )
    nodes["7"] = _node(
        "SaveImage",
        {
            "images": _ref("6", 0),
            "filename_prefix": "s4f3_sd15",
        },
    )
    return nodes


def _build_sdxl_nodes(
    prompt: str,
    preset: QualityPreset,
    width: int,
    height: int,
    model_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build SDXL txt2img nodes with dual CLIP encoders.

    Args:
        prompt: Positive prompt text.
        preset: Quality preset.
        width: Output image width.
        height: Output image height.
        model_name: Checkpoint filename.
        seed: Random seed.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("CheckpointLoaderSimple", {"ckpt_name": model_name})
    nodes["2"] = _node(
        "CLIPTextEncodeSDXL",
        {
            "text_g": prompt,
            "text_l": prompt,
            "clip": _ref("1", 1),
            "width": width,
            "height": height,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": width,
            "target_height": height,
        },
    )
    nodes["3"] = _node(
        "CLIPTextEncodeSDXL",
        {
            "text_g": preset.negative_template,
            "text_l": preset.negative_template,
            "clip": _ref("1", 1),
            "width": width,
            "height": height,
            "crop_w": 0,
            "crop_h": 0,
            "target_width": width,
            "target_height": height,
        },
    )
    nodes["4"] = _node(
        "EmptyLatentImage",
        {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    )
    nodes["5"] = _node(
        "KSampler",
        {
            "model": _ref("1", 0),
            "positive": _ref("2", 0),
            "negative": _ref("3", 0),
            "latent_image": _ref("4", 0),
            "seed": seed,
            "steps": preset.steps,
            "cfg": preset.cfg,
            "sampler_name": preset.sampler,
            "scheduler": preset.scheduler,
            "denoise": 1.0,
        },
    )
    nodes["6"] = _node(
        "VAEDecode",
        {
            "samples": _ref("5", 0),
            "vae": _ref("1", 2),
        },
    )
    nodes["7"] = _node(
        "SaveImage",
        {
            "images": _ref("6", 0),
            "filename_prefix": "s4f3_sdxl",
        },
    )
    return nodes


def _build_flux_nodes(
    prompt: str,
    preset: QualityPreset,
    width: int,
    height: int,
    model_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build Flux txt2img nodes with T5 encoder and FluxGuidance.

    Flux models use no negative prompt and a fixed CFG of 1.0 with
    guidance provided via the FluxGuidance node.

    Args:
        prompt: Positive prompt text.
        preset: Quality preset.
        width: Output image width.
        height: Output image height.
        model_name: Checkpoint filename.
        seed: Random seed.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("CheckpointLoaderSimple", {"ckpt_name": model_name})
    nodes["2"] = _node(
        "CLIPTextEncode",
        {
            "text": prompt,
            "clip": _ref("1", 1),
        },
    )
    nodes["3"] = _node(
        "FluxGuidance",
        {
            "conditioning": _ref("2", 0),
            "guidance": 3.5,
        },
    )
    nodes["4"] = _node(
        "EmptyLatentImage",
        {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    )
    nodes["5"] = _node(
        "KSampler",
        {
            "model": _ref("1", 0),
            "positive": _ref("3", 0),
            "negative": _ref("2", 0),
            "latent_image": _ref("4", 0),
            "seed": seed,
            "steps": preset.steps,
            "cfg": preset.cfg,
            "sampler_name": preset.sampler,
            "scheduler": preset.scheduler,
            "denoise": 1.0,
        },
    )
    nodes["6"] = _node(
        "VAEDecode",
        {
            "samples": _ref("5", 0),
            "vae": _ref("1", 2),
        },
    )
    nodes["7"] = _node(
        "SaveImage",
        {
            "images": _ref("6", 0),
            "filename_prefix": "s4f3_flux",
        },
    )
    return nodes


def _build_sd3_nodes(
    prompt: str,
    preset: QualityPreset,
    width: int,
    height: int,
    model_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build SD3 txt2img nodes with triple CLIP encoding.

    Args:
        prompt: Positive prompt text.
        preset: Quality preset.
        width: Output image width.
        height: Output image height.
        model_name: Checkpoint filename.
        seed: Random seed.

    Returns:
        Dict of node_id -> node dict.
    """
    nodes: dict[str, dict[str, Any]] = {}
    nodes["1"] = _node("CheckpointLoaderSimple", {"ckpt_name": model_name})
    nodes["2"] = _node(
        "CLIPTextEncodeSD3",
        {
            "text": prompt,
            "clip": _ref("1", 1),
            "clip_l": True,
            "clip_g": True,
            "t5xxl": True,
        },
    )
    nodes["3"] = _node(
        "CLIPTextEncodeSD3",
        {
            "text": preset.negative_template,
            "clip": _ref("1", 1),
            "clip_l": True,
            "clip_g": True,
            "t5xxl": True,
        },
    )
    nodes["4"] = _node(
        "EmptyLatentImage",
        {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    )
    nodes["5"] = _node(
        "KSampler",
        {
            "model": _ref("1", 0),
            "positive": _ref("2", 0),
            "negative": _ref("3", 0),
            "latent_image": _ref("4", 0),
            "seed": seed,
            "steps": preset.steps,
            "cfg": preset.cfg,
            "sampler_name": preset.sampler,
            "scheduler": preset.scheduler,
            "denoise": 1.0,
        },
    )
    nodes["6"] = _node(
        "VAEDecode",
        {
            "samples": _ref("5", 0),
            "vae": _ref("1", 2),
        },
    )
    nodes["7"] = _node(
        "SaveImage",
        {
            "images": _ref("6", 0),
            "filename_prefix": "s4f3_sd3",
        },
    )
    return nodes


# Map from model family to builder function.
_BUILDERS = {
    "sd15": _build_sd15_nodes,
    "sdxl": _build_sdxl_nodes,
    "sdxl_turbo": _build_sdxl_nodes,
    "pony": _build_sdxl_nodes,
    "flux_dev": _build_flux_nodes,
    "flux_schnell": _build_flux_nodes,
    "flux2_dev": _build_flux_nodes,
    "flux2_klein": _build_flux_nodes,
    "sd3": _build_sd3_nodes,
    "cascade": _build_sd15_nodes,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WorkflowBuilder:
    """Model-aware ComfyUI workflow builder.

    Generates workflow dicts in ComfyUI API format using quality presets
    as defaults. Individual parameters can be overridden via keyword args.
    """

    def build_txt2img(
        self,
        prompt: str,
        model: str,
        width: int = 1024,
        height: int = 1024,
        *,
        seed: int = 0,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build a txt2img workflow for a given model checkpoint.

        Args:
            prompt: Positive prompt text.
            model: Checkpoint filename (used for family detection).
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Random seed for generation.
            **overrides: Override preset values (``steps``, ``cfg``,
                ``sampler``, ``scheduler``).

        Returns:
            ComfyUI API-format workflow dict.
        """
        family = detect_model_family(model)
        preset = get_preset(family)
        preset = _apply_overrides(preset, overrides)
        builder_fn = _BUILDERS.get(family, _build_sd15_nodes)
        nodes = builder_fn(prompt, preset, width, height, model, seed)
        return {"prompt": nodes}

    def build_img2img(
        self,
        prompt: str,
        model: str,
        image_path: str,
        denoise: float = 0.7,
        *,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build an img2img workflow for a given model checkpoint.

        Loads a source image, encodes it to latent space, and runs the
        sampler with reduced denoise strength.

        Args:
            prompt: Positive prompt text.
            model: Checkpoint filename.
            image_path: Path to the source image file.
            denoise: Denoise strength (0.0-1.0, lower preserves more).
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Random seed for generation.
            **overrides: Override preset values.

        Returns:
            ComfyUI API-format workflow dict.
        """
        family = detect_model_family(model)
        preset = get_preset(family)
        preset = _apply_overrides(preset, overrides)

        nodes: dict[str, dict[str, Any]] = {}
        nodes["1"] = _node("CheckpointLoaderSimple", {"ckpt_name": model})
        nodes["2"] = _node(
            "CLIPTextEncode",
            {
                "text": prompt,
                "clip": _ref("1", 1),
            },
        )
        neg_text = preset.negative_template if not preset.no_negative_prompt else ""
        nodes["3"] = _node(
            "CLIPTextEncode",
            {
                "text": neg_text,
                "clip": _ref("1", 1),
            },
        )
        nodes["4"] = _node("LoadImage", {"image": image_path})
        nodes["5"] = _node(
            "VAEEncode",
            {
                "pixels": _ref("4", 0),
                "vae": _ref("1", 2),
            },
        )
        nodes["6"] = _node(
            "KSampler",
            {
                "model": _ref("1", 0),
                "positive": _ref("2", 0),
                "negative": _ref("3", 0),
                "latent_image": _ref("5", 0),
                "seed": seed,
                "steps": preset.steps,
                "cfg": preset.cfg,
                "sampler_name": preset.sampler,
                "scheduler": preset.scheduler,
                "denoise": denoise,
            },
        )
        nodes["7"] = _node(
            "VAEDecode",
            {
                "samples": _ref("6", 0),
                "vae": _ref("1", 2),
            },
        )
        nodes["8"] = _node(
            "SaveImage",
            {
                "images": _ref("7", 0),
                "filename_prefix": "s4f3_img2img",
            },
        )
        return {"prompt": nodes}


def _apply_overrides(preset: QualityPreset, overrides: dict[str, Any]) -> QualityPreset:
    """Apply user overrides to a preset, returning a new instance.

    Args:
        preset: Base quality preset.
        overrides: Dict of field names to override values.

    Returns:
        New ``QualityPreset`` with overrides applied.
    """
    if not overrides:
        return preset
    fields = {f.name: getattr(preset, f.name) for f in preset.__dataclass_fields__.values()}
    for key, value in overrides.items():
        if key in fields:
            fields[key] = value
    return QualityPreset(**fields)
