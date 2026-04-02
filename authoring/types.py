"""Pydantic models for ComfyUI custom node specifications.

Provides type-safe representations of node inputs, outputs, widgets,
and complete node specifications. Includes a simple NL parser for
converting natural language descriptions into NodeSpec objects.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WidgetType(str, Enum):
    """Supported ComfyUI widget/data types."""

    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    COMBO = "COMBO"
    IMAGE = "IMAGE"
    MASK = "MASK"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"
    LATENT = "LATENT"


class WidgetConfig(BaseModel):
    """Configuration for a node widget.

    Attributes:
        widget_type: The ComfyUI type for this widget.
        default: Default value for the widget.
        min_value: Minimum allowed value (numeric types).
        max_value: Maximum allowed value (numeric types).
        step: Step increment for numeric widgets.
        choices: List of valid choices for COMBO widgets.
        multiline: Whether STRING widget supports multiple lines.
        hidden: Whether this widget is hidden by default.
    """

    widget_type: WidgetType
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    multiline: bool = False
    hidden: bool = False


class InputSpec(BaseModel):
    """Specification for a single node input.

    Attributes:
        name: Parameter name used in code.
        widget: Widget configuration for this input.
        required: Whether this input is mandatory.
        tooltip: Help text displayed on hover.
    """

    name: str
    widget: WidgetConfig
    required: bool = True
    tooltip: str = ""


class OutputSpec(BaseModel):
    """Specification for a single node output.

    Attributes:
        name: Output slot name.
        type: ComfyUI data type for the output.
        tooltip: Help text displayed on hover.
    """

    name: str
    type: WidgetType
    tooltip: str = ""


class NodeSpec(BaseModel):
    """Complete specification for a ComfyUI custom node.

    Attributes:
        name: Internal class name for the node.
        display_name: Human-readable name shown in the UI.
        category: Category path in the node menu.
        description: Detailed description of the node's purpose.
        inputs: List of input specifications.
        outputs: List of output specifications.
        function_name: Name of the execution method.
        is_output_node: Whether this node produces final output.
        v3_format: Whether to generate V3 (stateless classmethod) format.
    """

    name: str = Field(..., min_length=1, max_length=128)
    display_name: str = ""
    category: str = "custom"
    description: str = ""
    inputs: list[InputSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    function_name: str = "execute"
    is_output_node: bool = False
    v3_format: bool = True

    def model_post_init(self, __context: Any) -> None:
        """Auto-generate display_name from name if not provided."""
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


# --- Natural Language Parser ---

# Maps common words to WidgetType
_TYPE_KEYWORDS: dict[str, WidgetType] = {
    "image": WidgetType.IMAGE,
    "images": WidgetType.IMAGE,
    "picture": WidgetType.IMAGE,
    "photo": WidgetType.IMAGE,
    "mask": WidgetType.MASK,
    "model": WidgetType.MODEL,
    "clip": WidgetType.CLIP,
    "vae": WidgetType.VAE,
    "conditioning": WidgetType.CONDITIONING,
    "latent": WidgetType.LATENT,
    "text": WidgetType.STRING,
    "string": WidgetType.STRING,
    "prompt": WidgetType.STRING,
    "name": WidgetType.STRING,
    "label": WidgetType.STRING,
    "number": WidgetType.FLOAT,
    "float": WidgetType.FLOAT,
    "integer": WidgetType.INT,
    "int": WidgetType.INT,
    "count": WidgetType.INT,
    "boolean": WidgetType.BOOLEAN,
    "bool": WidgetType.BOOLEAN,
    "toggle": WidgetType.BOOLEAN,
    "flag": WidgetType.BOOLEAN,
    "enabled": WidgetType.BOOLEAN,
}

# Patterns for extracting numeric ranges like (0-100), (0.0-1.0)
_RANGE_PATTERN = re.compile(r"\((\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\)")

# Patterns for input/output sections
_INPUT_PATTERN = re.compile(
    r"(?:takes?|accepts?|receives?|inputs?|with)\s+(.+?)(?:\s+and\s+(?:outputs?|returns?|produces?)|$)",
    re.IGNORECASE,
)
_OUTPUT_PATTERN = re.compile(
    r"(?:outputs?|returns?|produces?|generates?|emits?)\s+(.+)",
    re.IGNORECASE,
)


def _infer_type_from_token(token: str) -> WidgetType:
    """Infer a WidgetType from a single descriptive token.

    Args:
        token: A lowercase word describing a parameter.

    Returns:
        Best-matching WidgetType, defaults to FLOAT.
    """
    token_lower = token.strip().lower()
    for keyword, wtype in _TYPE_KEYWORDS.items():
        if keyword in token_lower:
            return wtype
    return WidgetType.FLOAT


def _parse_parameter(text: str) -> tuple[str, WidgetConfig]:
    """Parse a single parameter description into name and config.

    Args:
        text: A fragment like "a blur strength (0-100)" or "an image".

    Returns:
        Tuple of (parameter_name, WidgetConfig).
    """
    text = text.strip().strip(",").strip()

    # Extract numeric range if present
    range_match = _RANGE_PATTERN.search(text)
    min_val = None
    max_val = None
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        text = text[: range_match.start()] + text[range_match.end() :]

    # Remove articles
    cleaned = re.sub(r"\b(a|an|the)\b", "", text, flags=re.IGNORECASE).strip()

    # Extract meaningful words
    words = [w for w in cleaned.split() if w.isalpha()]
    if not words:
        words = ["parameter"]

    # Build parameter name from words
    param_name = "_".join(words).lower()

    # Infer type from the words
    widget_type = WidgetType.FLOAT
    for word in words:
        inferred = _infer_type_from_token(word)
        if inferred != WidgetType.FLOAT:
            widget_type = inferred
            break

    # If we found a range and the type is still generic, use FLOAT or INT
    if range_match and widget_type == WidgetType.FLOAT:
        if min_val is not None and max_val is not None:
            if min_val == int(min_val) and max_val == int(max_val):
                widget_type = WidgetType.INT

    config = WidgetConfig(
        widget_type=widget_type,
        min_value=min_val,
        max_value=max_val,
    )

    return param_name, config


def _split_items(text: str) -> list[str]:
    """Split a comma/and-separated list into individual items.

    Args:
        text: Text like "an image, a blur strength (0-100), and a mask".

    Returns:
        List of individual parameter descriptions.
    """
    # Split on "," and "and"
    parts = re.split(r"\s*,\s*|\s+and\s+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_natural_language(description: str) -> NodeSpec:
    """Convert a natural language description into a NodeSpec.

    Parses simple NL descriptions of node functionality, extracting
    inputs and outputs from pattern matching.

    Args:
        description: Natural language like "create a node that takes
            an image and a blur strength (0-100) and outputs a
            blurred image".

    Returns:
        A populated NodeSpec with inferred inputs and outputs.

    Examples:
        >>> spec = parse_natural_language(
        ...     "create a node that takes an image and a blur "
        ...     "strength (0-100) and outputs a blurred image"
        ... )
        >>> len(spec.inputs)
        2
        >>> spec.outputs[0].type
        <WidgetType.IMAGE: 'IMAGE'>
    """
    # Extract a node name from the description
    name_match = re.search(
        r"(?:node|block)\s+(?:called|named)\s+['\"]?(\w+)['\"]?",
        description,
        re.IGNORECASE,
    )
    if name_match:
        node_name = name_match.group(1)
    else:
        # Generate name from key action words
        action_words = re.findall(
            r"\b(blur|resize|crop|filter|mix|blend|merge|split|invert|sharpen|denoise|upscale|transform|convert|color|adjust|mask|composite)\b",
            description,
            re.IGNORECASE,
        )
        if action_words:
            node_name = "_".join(w.lower() for w in action_words[:3]) + "_node"
        else:
            node_name = "custom_node"

    inputs: list[InputSpec] = []
    outputs: list[OutputSpec] = []

    # Parse inputs
    input_match = _INPUT_PATTERN.search(description)
    if input_match:
        input_text = input_match.group(1)
        for item in _split_items(input_text):
            param_name, config = _parse_parameter(item)
            inputs.append(InputSpec(name=param_name, widget=config))

    # Parse outputs
    output_match = _OUTPUT_PATTERN.search(description)
    if output_match:
        output_text = output_match.group(1)
        for item in _split_items(output_text):
            param_name, config = _parse_parameter(item)
            outputs.append(OutputSpec(name=param_name, type=config.widget_type))

    return NodeSpec(
        name=node_name,
        description=description.strip(),
        inputs=inputs,
        outputs=outputs,
    )
