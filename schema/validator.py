"""
ComfyUI Workflow Validator

Main validator class for ComfyUI workflow JSON files.
Supports both Version 0.x (legacy) and Version 1 (current) formats.
"""

import json
from pathlib import Path
from typing import Any

from .types import (
    SchemaValidationResult,
    SchemaVersion,
    WorkflowValidationReport,
    is_valid_uuid,
)
from .validators_common import validate_node_link_consistency, validate_nodes
from .validators_v0 import validate_v0
from .validators_v1 import validate_v1


class ComfyUIWorkflowValidator:
    """
    Validates ComfyUI workflow JSON files against the official schema.

    Supports two schema versions:

    Version 0 (Legacy):
    - version: 0.x (number, e.g., 0.4)
    - id: simple string or UUID
    - last_node_id, last_link_id: numbers
    - links: array of 6-element arrays OR array of objects

    Version 1 (Current - requires Zod validation):
    - version: 1 (integer)
    - id: UUID string (required)
    - state: object with lastGroupId, lastNodeId, lastLinkId, lastRerouteId
    - links: array of SerialisableLLink objects
    """

    def __init__(self, strict_mode: bool = False, target_version: int | None = None):
        """
        Initialize the validator.

        Args:
            strict_mode: If True, warnings are treated as errors
            target_version: If set, validate against specific version (0 or 1)
        """
        self.strict_mode = strict_mode
        self.target_version = target_version

    def detect_version(self, data: dict[str, Any]) -> int:
        """
        Detect the schema version from workflow data.

        Returns:
            SchemaVersion.V0 or SchemaVersion.V1
        """
        version = data.get("version")

        # Version 1 uses integer 1
        if version == 1:
            return SchemaVersion.V1

        # Version 0.x uses float/decimal
        if isinstance(version, (int, float)) and version < 1:
            return SchemaVersion.V0

        # Check for version 1 indicators
        has_state = "state" in data and isinstance(data.get("state"), dict)
        has_uuid_id = is_valid_uuid(str(data.get("id", "")))

        if has_state or has_uuid_id:
            return SchemaVersion.V1

        # Default to V0 for legacy compatibility
        return SchemaVersion.V0

    def validate_file(self, file_path: str | Path) -> WorkflowValidationReport:
        """
        Validate a workflow JSON file.

        Args:
            file_path: Path to the workflow JSON file

        Returns:
            WorkflowValidationReport with validation results
        """
        file_path = Path(file_path)
        report = WorkflowValidationReport(file_path=str(file_path), valid=True)

        if not file_path.exists():
            report.valid = False
            report.add_result(
                SchemaValidationResult(
                    valid=False, field="file", message=f"File not found: {file_path}", severity="error"
                )
            )
            return report

        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            report.valid = False
            report.add_result(
                SchemaValidationResult(valid=False, field="json", message=f"Invalid JSON: {e}", severity="error")
            )
            return report

        return self.validate_dict(data, str(file_path))

    def validate_dict(self, data: dict[str, Any], source: str = "workflow") -> WorkflowValidationReport:
        """
        Validate a workflow dictionary.

        Args:
            data: The workflow data as a dictionary
            source: Source identifier for the report

        Returns:
            WorkflowValidationReport with validation results
        """
        report = WorkflowValidationReport(file_path=source, valid=True)

        # Detect version
        detected_version = self.detect_version(data)
        report.detected_version = detected_version

        # Use target version if specified, otherwise use detected
        version = self.target_version if self.target_version is not None else detected_version

        report.add_result(
            SchemaValidationResult(
                valid=True,
                field="version_detection",
                message=f"Detected schema version: {detected_version}, validating as: {version}",
                severity="info",
            )
        )

        # Run version-specific validations
        if version == SchemaVersion.V1:
            validate_v1(data, report)
        else:
            validate_v0(data, report)

        # Common validations
        validate_nodes(data, report, version)
        validate_node_link_consistency(data, report, version)

        # Set overall validity
        if report.errors or (self.strict_mode and report.warnings):
            report.valid = False

        return report

    # Backward compatibility methods - delegate to extracted functions
    def _validate_v1(self, data: dict[str, Any], report: WorkflowValidationReport) -> None:
        """Validate Version 1 specific requirements."""
        validate_v1(data, report)

    def _validate_v0(self, data: dict[str, Any], report: WorkflowValidationReport) -> None:
        """Validate Version 0 specific requirements."""
        validate_v0(data, report)

    def _validate_nodes(self, data: dict[str, Any], report: WorkflowValidationReport, version: int) -> None:
        """Validate the nodes array."""
        validate_nodes(data, report, version)

    def _validate_node_link_consistency(
        self, data: dict[str, Any], report: WorkflowValidationReport, version: int
    ) -> None:
        """Validate node-link consistency."""
        validate_node_link_consistency(data, report, version)


# CONVENIENCE FUNCTIONS


def validate_workflow_file(
    file_path: str | Path, strict: bool = False, target_version: int | None = None
) -> WorkflowValidationReport:
    """
    Convenience function to validate a workflow file.

    Args:
        file_path: Path to the workflow JSON file
        strict: If True, treat warnings as errors
        target_version: If set, validate against specific version (0 or 1)

    Returns:
        WorkflowValidationReport with validation results
    """
    validator = ComfyUIWorkflowValidator(strict_mode=strict, target_version=target_version)
    return validator.validate_file(file_path)


def validate_workflow_dict(
    data: dict[str, Any], strict: bool = False, target_version: int | None = None
) -> WorkflowValidationReport:
    """
    Convenience function to validate a workflow dictionary.

    Args:
        data: The workflow data as a dictionary
        strict: If True, treat warnings as errors
        target_version: If set, validate against specific version (0 or 1)

    Returns:
        WorkflowValidationReport with validation results
    """
    validator = ComfyUIWorkflowValidator(strict_mode=strict, target_version=target_version)
    return validator.validate_dict(data)
