"""ComfyUI custom node authoring engine.

Public API for creating, validating, and testing ComfyUI custom nodes.
These functions will be registered as MCP tools by Phase 8.

Example usage::

    from comfyui_node_authoring import create_custom_node, validate_node

    result = create_custom_node({
        "name": "blur_image",
        "category": "image/filter",
        "inputs": [
            {"name": "image", "widget": {"widget_type": "IMAGE"}},
            {"name": "radius", "widget": {
                "widget_type": "FLOAT",
                "min_value": 0, "max_value": 100,
            }},
        ],
        "outputs": [{"name": "image", "type": "IMAGE"}],
    })

    validation = validate_node(result.file_path)
    print(validation.passed)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .composite import (
    create_composite_spec,
    generate_composite_node,
    split_node_spec,
    validate_composite_outputs,
)
from .generator import GenerationResult, generate_node_code, write_node_package
from .layout import LayoutResult, optimize_layout
from .test_harness import ValidationResult
from .test_harness import validate_node as _validate_source
from .types import (
    InputSpec,
    NodeSpec,
    OutputSpec,
    WidgetConfig,
    WidgetType,
    parse_natural_language,
)


def create_custom_node(
    spec_json: dict[str, Any] | str,
    output_dir: Path | str | None = None,
) -> GenerationResult:
    """Create a complete ComfyUI custom node from a JSON specification.

    Parses the spec, generates code, validates it, and optionally
    writes it to disk.

    Args:
        spec_json: Node specification as a dict or JSON string.
        output_dir: Directory to write the node package. Uses a temp
            directory if not provided.

    Returns:
        GenerationResult with generated source and validation status.

    Raises:
        ValueError: If the spec is invalid.
    """
    if isinstance(spec_json, str):
        spec_json = json.loads(spec_json)

    spec = NodeSpec.model_validate(spec_json)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="comfyui_node_"))
    else:
        output_dir = Path(output_dir)

    return write_node_package(spec, output_dir)


def validate_node(node_path: Path | str) -> ValidationResult:
    """Run the full test harness on a generated node file.

    Args:
        node_path: Path to a generated Python node file.

    Returns:
        ValidationResult with pass/fail and detailed checks.
    """
    path = Path(node_path)
    source = path.read_text(encoding="utf-8")
    return _validate_source(source)


__all__ = [
    # Public API
    "create_custom_node",
    "validate_node",
    # Types
    "InputSpec",
    "LayoutResult",
    "NodeSpec",
    "OutputSpec",
    "WidgetConfig",
    "WidgetType",
    # Generator
    "GenerationResult",
    "generate_node_code",
    "write_node_package",
    # Test harness
    "ValidationResult",
    # Layout
    "optimize_layout",
    # Composite
    "create_composite_spec",
    "generate_composite_node",
    "split_node_spec",
    "validate_composite_outputs",
    # NL Parser
    "parse_natural_language",
]
