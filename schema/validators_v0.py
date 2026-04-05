"""
ComfyUI Workflow Validator - Version 0 Validations.

Contains all validation logic specific to V0 (legacy) schema format.
V0 uses last_node_id/last_link_id and supports links as arrays or objects.
"""

from typing import TYPE_CHECKING, Any

from .types import SchemaValidationResult

if TYPE_CHECKING:
    from .types import WorkflowValidationReport


def validate_v0(data: dict[str, Any], report: "WorkflowValidationReport") -> None:
    """Validate Version 0 specific requirements."""
    validate_v0_version(data, report)
    validate_v0_id_trackers(data, report)
    validate_v0_links(data, report)


def validate_v0_version(
    data: dict[str, Any], report: "WorkflowValidationReport"
) -> None:
    """Validate version field for V0."""
    version = data.get("version")
    if version is None:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="version",
                message="Missing required 'version' field",
                severity="error",
                details={"fix": "Add 'version': 0.4 for V0 or 'version': 1 for V1"},
            )
        )
    elif not isinstance(version, (int, float)):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="version",
                message=f"'version' must be a number, got: {type(version).__name__}",
                severity="error",
            )
        )


def validate_v0_id_trackers(
    data: dict[str, Any], report: "WorkflowValidationReport"
) -> None:
    """Validate last_node_id and last_link_id for V0."""
    # These are recommended but not strictly required in V0
    if "last_node_id" not in data:
        report.add_result(
            SchemaValidationResult(
                valid=True,
                field="last_node_id",
                message="Missing recommended 'last_node_id' field",
                severity="warning",
            )
        )

    if "last_link_id" not in data:
        report.add_result(
            SchemaValidationResult(
                valid=True,
                field="last_link_id",
                message="Missing recommended 'last_link_id' field",
                severity="warning",
            )
        )


def validate_v0_links(data: dict[str, Any], report: "WorkflowValidationReport") -> None:
    """Validate links array for V0 (can be arrays or objects)."""
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

    link_ids: set[Any] = set()
    for i, link in enumerate(links):
        # V0 supports both array and object formats
        if isinstance(link, list):
            # Array format: [id, origin_id, origin_slot, target_id, target_slot, type]
            if len(link) != 6:
                report.add_result(
                    SchemaValidationResult(
                        valid=False,
                        field=f"links[{i}]",
                        message=f"Array link must have 6 elements, got: {len(link)}",
                        severity="error",
                    )
                )
                continue
            link_id = link[0]
        elif isinstance(link, dict):
            # Object format
            link_id = link.get("id")
        else:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"links[{i}]",
                    message=f"Link must be array or object, got: {type(link).__name__}",
                    severity="error",
                )
            )
            continue

        # Check for duplicate IDs
        if link_id is not None:
            if link_id in link_ids:
                report.add_result(
                    SchemaValidationResult(
                        valid=False,
                        field=f"links[{i}]",
                        message=f"Duplicate link ID: {link_id}",
                        severity="error",
                    )
                )
            link_ids.add(link_id)
