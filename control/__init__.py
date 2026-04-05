"""ComfyUI Extended Control Surface.

Provides high-level functions for workflow template management,
model operations, batch generation, and server lifecycle control.

Submodules:
    templates: Workflow template CRUD and variable rendering.
    model_manager: Model listing, architecture detection, merge, checksum.
    batch: Parameter/seed sweeping with concurrent queue management.
    server_lifecycle: ComfyUI server start/stop/restart/health.

Usage:
    from lib.comfyui_control import manage_templates, manage_models
    from lib.comfyui_control import batch_generate, server_control
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .batch import BatchConfig, BatchGenerator, BatchResult
from .model_manager import ModelInfo, ModelManager
from .server_lifecycle import ServerConfig, ServerLifecycle
from .templates import TemplateManager, WorkflowTemplate

__all__ = [
    "BatchConfig",
    "BatchGenerator",
    "BatchResult",
    "ModelInfo",
    "ModelManager",
    "ServerConfig",
    "ServerLifecycle",
    "TemplateManager",
    "WorkflowTemplate",
    "batch_generate",
    "manage_models",
    "manage_templates",
    "server_control",
]


def manage_templates(
    action: str,
    template_dir: str = "",
    **kwargs: Any,
) -> Any:
    """CRUD entry point for workflow templates.

    Args:
        action: One of list, get, import, export, render, save, delete.
        template_dir: Directory for template storage.
        **kwargs: Action-specific keyword arguments.

    Returns:
        Action-dependent result (template, list, path, or rendered dict).

    Raises:
        ValueError: If action is not recognised.
    """
    mgr = TemplateManager(template_dir or Path.cwd() / "templates")
    actions = {
        "list": lambda: mgr.list_templates(category=kwargs.get("category")),
        "get": lambda: mgr.get_template(kwargs["name"]),
        "import": lambda: mgr.import_workflow(
            kwargs["json_path"],
            kwargs["name"],
            kwargs["category"],
            description=kwargs.get("description", ""),
        ),
        "export": lambda: mgr.export_template(kwargs["name"], kwargs["output_path"]),
        "render": lambda: mgr.render_template(
            kwargs["name"],
            kwargs.get("variables"),
        ),
        "save": lambda: mgr.save_template(kwargs["template"]),
        "delete": lambda: mgr.delete_template(kwargs["name"]),
    }
    if action not in actions:
        msg = f"Unknown template action '{action}'. Valid: {sorted(actions)}"
        raise ValueError(msg)
    return actions[action]()


def manage_models(
    action: str,
    models_dir: str = "",
    **kwargs: Any,
) -> Any:
    """Entry point for model management operations.

    Args:
        action: One of list, info, download, detect, convert, merge, checksum.
        models_dir: Root directory containing model files.
        **kwargs: Action-specific keyword arguments.

    Returns:
        Action-dependent result.

    Raises:
        ValueError: If action is not recognised.
    """
    mgr = ModelManager(models_dir or Path.cwd() / "models")
    actions = {
        "list": lambda: mgr.list_models(),
        "info": lambda: mgr.get_model_info(kwargs["name"]),
        "detect": lambda: mgr.detect_architecture(kwargs["model_path"]),
        "convert": lambda: mgr.convert_precision(
            kwargs["model_path"],
            kwargs.get("target_dtype", "fp16"),
            output_name=kwargs.get("output_name", ""),
        ),
        "merge": lambda: mgr.merge_models(
            kwargs["model_a"],
            kwargs["model_b"],
            alpha=kwargs.get("alpha", 0.5),
            output_name=kwargs.get("output_name", ""),
        ),
        "checksum": lambda: mgr.verify_checksum(kwargs["model_path"]),
    }
    if action not in actions:
        msg = f"Unknown model action '{action}'. Valid: {sorted(actions)}"
        raise ValueError(msg)
    return actions[action]()


async def batch_generate(
    config: BatchConfig,
    comfyui_url: str = "http://127.0.0.1:8188",
    **kwargs: Any,
) -> BatchResult:
    """Launch a batch generation run with progress tracking.

    Args:
        config: Batch configuration with prompts and sweep params.
        comfyui_url: ComfyUI server URL.
        **kwargs: Passed to BatchGenerator.run_batch.

    Returns:
        Aggregated BatchResult.
    """
    gen = BatchGenerator(comfyui_url)
    return await gen.run_batch(config, **kwargs)


async def server_control(
    action: str,
    comfyui_path: str | None = None,
    **kwargs: Any,
) -> Any:
    """Control the ComfyUI server lifecycle.

    Args:
        action: One of start, stop, restart, status, health.
        comfyui_path: Path to ComfyUI installation.
        **kwargs: Action-specific keyword arguments.

    Returns:
        Action-dependent result (bool, dict, or int).

    Raises:
        ValueError: If action is not recognised.
    """
    server = ServerLifecycle(comfyui_path)
    if action == "start":
        config = kwargs.get("config")
        return await server.start(config)
    if action == "stop":
        return await server.stop(kwargs.get("timeout", 10.0))
    if action == "restart":
        return await server.restart(kwargs.get("config"))
    if action == "status":
        return await server.is_running()
    if action == "health":
        return await server.health_check()
    msg = (
        f"Unknown server action '{action}'. Valid: start, stop, restart, status, health"
    )
    raise ValueError(msg)
