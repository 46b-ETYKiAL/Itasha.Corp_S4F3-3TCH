"""ComfyUI Workflow Validator - Version 1 Validations.

Contains all validation logic specific to V1 (current) schema format.
V1 requires UUID id, state object, and links as objects.

Extended to validate V3 node metadata and VIDEO/AUDIO output types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import (
    KNOWN_OUTPUT_TYPES,
    SchemaValidationResult,
    generate_uuid,
    is_known_output_type,
    is_valid_uuid,
    parse_gguf_reference,
    parse_v3_metadata,
)

if TYPE_CHECKING:
    from .types import WorkflowValidationReport


def validate_v1(data: dict[str, Any], report: WorkflowValidationReport) -> None:
    """Validate Version 1 specific requirements."""
    validate_v1_version(data, report)
    validate_v1_id(data, report)
    validate_v1_state(data, report)
    validate_v1_links(data, report)
    validate_v1_output_types(data, report)
    validate_v1_v3_metadata(data, report)
    validate_v1_gguf_references(data, report)


def validate_v1_version(data: dict[str, Any], report: WorkflowValidationReport) -> None:
    """Validate version field for V1."""
    version = data.get("version")
    if version != 1:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="version",
                message=f"Version 1 requires 'version': 1 (integer), got: {version}",
                severity="error",
                details={"expected": 1, "got": version},
            )
        )


def validate_v1_id(data: dict[str, Any], report: WorkflowValidationReport) -> None:
    """Validate UUID id field for V1."""
    workflow_id = data.get("id")

    if workflow_id is None:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="id",
                message="Version 1 requires 'id' field with UUID format",
                severity="error",
                details={"fix": f"Add 'id': '{generate_uuid()}'"},
            )
        )
        return

    if not is_valid_uuid(str(workflow_id)):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="id",
                message=f"Invalid UUID format for 'id': {workflow_id}",
                severity="error",
                details={
                    "got": workflow_id,
                    "expected": "UUID format (e.g., 9ae6082b-c7f4-433c-9971-7a8f65a3ea65)",
                    "fix": f"Replace with '{generate_uuid()}'",
                },
            )
        )


def validate_v1_state(data: dict[str, Any], report: WorkflowValidationReport) -> None:
    """Validate state object for V1."""
    state = data.get("state")

    if state is None:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="state",
                message="Version 1 requires 'state' object",
                severity="error",
                details={
                    "fix": "Add 'state': {'lastGroupId': 0, 'lastNodeId': N, 'lastLinkId': M, 'lastRerouteId': 0}"
                },
            )
        )
        return

    if not isinstance(state, dict):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="state",
                message=f"'state' must be an object, got: {type(state).__name__}",
                severity="error",
            )
        )
        return

    # Check required state fields
    required_state_fields = ["lastNodeId", "lastLinkId"]

    for field_name in required_state_fields:
        if field_name not in state:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"state.{field_name}",
                    message=f"Missing required state field: {field_name}",
                    severity="error",
                )
            )
        elif not isinstance(state[field_name], (int, float)):
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"state.{field_name}",
                    message=f"state.{field_name} must be a number",
                    severity="error",
                )
            )


def validate_v1_links(data: dict[str, Any], report: WorkflowValidationReport) -> None:
    """Validate links array for V1 (must be objects, not arrays)."""
    links = data.get("links")

    if links is None:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="links",
                message="Missing required 'links' field",
                severity="error",
            )
        )
        return

    if not isinstance(links, list):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="links",
                message=f"'links' must be an array, got: {type(links).__name__}",
                severity="error",
            )
        )
        return

    # V1 requires links to be objects (SerialisableLLink)
    link_ids: set[Any] = set()
    for i, link in enumerate(links):
        if isinstance(link, list):
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"links[{i}]",
                    message="Version 1 requires links as objects, not arrays",
                    severity="error",
                    details={
                        "got": "array",
                        "expected": "object with {id, origin_id, origin_slot, target_id, target_slot, type}",
                        "fix": "Convert to object format",
                    },
                )
            )
            continue

        if not isinstance(link, dict):
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"links[{i}]",
                    message=f"Link must be an object, got: {type(link).__name__}",
                    severity="error",
                )
            )
            continue

        # Validate required link object fields
        required_fields = [
            "id",
            "origin_id",
            "origin_slot",
            "target_id",
            "target_slot",
            "type",
        ]
        for field_name in required_fields:
            if field_name not in link:
                report.add_result(
                    SchemaValidationResult(
                        valid=False,
                        field=f"links[{i}].{field_name}",
                        message=f"Link missing required field: {field_name}",
                        severity="error",
                    )
                )

        # Check for duplicate IDs
        link_id = link.get("id")
        if link_id is not None:
            if link_id in link_ids:
                report.add_result(
                    SchemaValidationResult(
                        valid=False,
                        field=f"links[{i}].id",
                        message=f"Duplicate link ID: {link_id}",
                        severity="error",
                    )
                )
            link_ids.add(link_id)


def validate_v1_output_types(
    data: dict[str, Any], report: WorkflowValidationReport
) -> None:
    """Validate output types in workflow nodes against known types.

    Accepts VIDEO, AUDIO, and other extended types added for V3
    workflows (Wan 2.2, LTX-Video, etc.).
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        outputs = node.get("outputs", [])
        if not isinstance(outputs, list):
            continue
        for j, output in enumerate(outputs):
            if not isinstance(output, dict):
                continue
            output_type = output.get("type", "")
            if output_type and not is_known_output_type(output_type):
                report.add_result(
                    SchemaValidationResult(
                        valid=True,
                        field=f"nodes[{i}].outputs[{j}].type",
                        message=(
                            f"Unknown output type '{output_type}'. Known types: {sorted(KNOWN_OUTPUT_TYPES)}"
                        ),
                        severity="warning",
                        details={"output_type": output_type},
                    )
                )


def validate_v1_v3_metadata(
    data: dict[str, Any], report: WorkflowValidationReport
) -> None:
    """Parse and validate V3 node metadata (informational).

    V3 metadata fields (api_version, caching_policy) are parsed
    but not required.  Invalid values generate info-level messages.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return

    valid_caching = {"always", "never", "auto"}
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        meta = parse_v3_metadata(node)
        if meta.is_v3:
            report.add_result(
                SchemaValidationResult(
                    valid=True,
                    field=f"nodes[{i}].properties.api_version",
                    message=f"V3 node detected (api_version={meta.api_version})",
                    severity="info",
                    details={
                        "api_version": meta.api_version,
                        "caching_policy": meta.caching_policy,
                    },
                )
            )
            if meta.caching_policy not in valid_caching:
                report.add_result(
                    SchemaValidationResult(
                        valid=True,
                        field=f"nodes[{i}].properties.caching_policy",
                        message=(
                            f"Unknown caching policy '{meta.caching_policy}'. Valid: {sorted(valid_caching)}"
                        ),
                        severity="warning",
                    )
                )


def validate_v1_gguf_references(
    data: dict[str, Any], report: WorkflowValidationReport
) -> None:
    """Detect GGUF model references in workflow nodes (informational).

    GGUF models require the ComfyUI-GGUF extension.  This validator
    flags their presence so compatibility checks can warn the user.
    """
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        ref = parse_gguf_reference(node)
        if ref is not None:
            report.add_result(
                SchemaValidationResult(
                    valid=True,
                    field=f"nodes[{i}].class_type",
                    message=(
                        f"GGUF model reference: {ref.model_name} "
                        f"(quant={ref.quantization_type or 'unknown'}, "
                        f"loader={ref.loader_node}). "
                        "Requires ComfyUI-GGUF extension."
                    ),
                    severity="info",
                    details={
                        "model_name": ref.model_name,
                        "quantization_type": ref.quantization_type,
                        "loader_node": ref.loader_node,
                    },
                )
            )
