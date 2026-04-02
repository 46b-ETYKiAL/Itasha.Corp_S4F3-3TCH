"""ComfyUI Workflow Schema Types.

Contains type definitions and data classes for workflow validation:
- SchemaVersion constants
- SchemaValidationResult and WorkflowValidationReport dataclasses
- UUID utilities
- Known output types including VIDEO and AUDIO
- V3 node metadata fields
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

# UUID pattern for validation
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def generate_uuid() -> str:
    """Generate a valid UUID v4 string."""
    return str(uuid.uuid4())


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    return bool(UUID_PATTERN.match(str(value)))


class SchemaVersion:
    """ComfyUI workflow schema versions."""

    V0 = 0  # Legacy: array links, numeric IDs
    V1 = 1  # Current: object links, UUID id, state object


@dataclass
class SchemaValidationResult:
    """Result of a schema validation check."""

    valid: bool
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    details: dict[str, Any] | None = None


@dataclass
class WorkflowValidationReport:
    """Complete validation report for a workflow file."""

    file_path: str
    valid: bool
    detected_version: int | None = None
    errors: list[SchemaValidationResult] = field(default_factory=list)
    warnings: list[SchemaValidationResult] = field(default_factory=list)
    info: list[SchemaValidationResult] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_result(self, result: SchemaValidationResult):
        """Add a validation result to the appropriate list."""
        if result.severity == "error":
            self.errors.append(result)
        elif result.severity == "warning":
            self.warnings.append(result)
        else:
            self.info.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "valid": self.valid,
            "detected_version": self.detected_version,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [{"field": e.field, "message": e.message, "details": e.details} for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message, "details": w.details} for w in self.warnings],
        }


# ---------------------------------------------------------------------------
# Known output types (extended for V3 + video/audio)
# ---------------------------------------------------------------------------

KNOWN_OUTPUT_TYPES: frozenset[str] = frozenset(
    {
        # Standard image types
        "IMAGE",
        "MASK",
        "LATENT",
        # Video types (Wan 2.2, LTX-Video, AnimateDiff, etc.)
        "VIDEO",
        "VIDEO_FRAMES",
        "ANIMATION",
        # Audio types
        "AUDIO",
        "AUDIO_WAVEFORM",
        # Model types
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONTROL_NET",
        # Data types
        "STRING",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "COMBO",
        # Special
        "PREVIEW",
        "*",
    }
)


@dataclass
class V3NodeMetadata:
    """V3 node API metadata fields (ComfyUI v0.18+).

    These fields are parsed from workflow JSON but are not required
    for validation (backwards-compatible).

    Attributes:
        api_version: V3 API version string (e.g. "1.0").
        caching_policy: Node caching policy ("always", "never", "auto").
        is_v3: Whether this node uses the V3 API.
    """

    api_version: str = ""
    caching_policy: str = "auto"
    is_v3: bool = False


@dataclass
class GGUFModelReference:
    """GGUF model reference in workflow configuration.

    Attributes:
        model_name: GGUF model filename.
        quantization_type: Quantization level (e.g. "q4_k_m", "q8_0").
        loader_node: Node class that loads the GGUF model.
    """

    model_name: str = ""
    quantization_type: str = ""
    loader_node: str = "UNETLoaderGGUF"


def is_known_output_type(output_type: str) -> bool:
    """Check whether an output type is in the known set.

    Args:
        output_type: Output type string to check.

    Returns:
        True if the type is known (including VIDEO, AUDIO).
    """
    return output_type.upper() in KNOWN_OUTPUT_TYPES


def parse_v3_metadata(node_data: dict[str, Any]) -> V3NodeMetadata:
    """Extract V3 metadata from a node definition dict.

    Args:
        node_data: Single node dict from a workflow JSON.

    Returns:
        V3NodeMetadata with parsed fields.
    """
    properties = node_data.get("properties", {})
    if not isinstance(properties, dict):
        return V3NodeMetadata()

    api_version = str(properties.get("api_version", ""))
    caching = str(properties.get("caching_policy", "auto"))
    is_v3 = api_version.startswith("1") or bool(properties.get("v3", False))

    return V3NodeMetadata(
        api_version=api_version,
        caching_policy=caching,
        is_v3=is_v3,
    )


def parse_gguf_reference(node_data: dict[str, Any]) -> GGUFModelReference | None:
    """Extract GGUF model reference from a node if it is a GGUF loader.

    Args:
        node_data: Single node dict from a workflow JSON.

    Returns:
        GGUFModelReference if the node loads a GGUF model, else None.
    """
    class_type = node_data.get("class_type", "")
    gguf_loaders = {"UNETLoaderGGUF", "DualCLIPLoaderGGUF", "CLIPLoaderGGUF"}

    if class_type not in gguf_loaders:
        return None

    inputs = node_data.get("inputs", {})
    if not isinstance(inputs, dict):
        return None

    model_name = str(inputs.get("unet_name", inputs.get("clip_name", "")))
    quant = ""
    if model_name:
        # Try to extract quant type from filename
        name_lower = model_name.lower()
        for qt in (
            "q2_k",
            "q3_k_s",
            "q3_k_m",
            "q3_k_l",
            "q4_k_s",
            "q4_k_m",
            "q4_0",
            "q4_1",
            "q5_k_s",
            "q5_k_m",
            "q5_0",
            "q5_1",
            "q6_k",
            "q8_0",
            "q8_1",
            "f16",
            "f32",
        ):
            if qt in name_lower:
                quant = qt
                break

    return GGUFModelReference(
        model_name=model_name,
        quantization_type=quant,
        loader_node=class_type,
    )
