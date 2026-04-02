"""
ComfyUI Workflow Schema Validator

Validates ComfyUI workflow JSON files against the official schema specification.
Supports both Version 0.x (legacy) and Version 1 (current) formats.

Submodules:
- types: Schema types, validation results, report classes
- validator: Main ComfyUIWorkflowValidator class
- validators_v0: Version 0 (legacy) validation functions
- validators_v1: Version 1 (current) validation functions
- validators_common: Shared validation logic (nodes, links)
- converters: V0 to V1 conversion utilities
- cli: Command-line interface

Official Schema Reference:
- https://docs.comfy.org/specs/workflow_json
- https://github.com/Comfy-Org/workflow_templates
- ComfyUI_frontend/src/lib/litegraph/src/types/serialisation.ts

Schema Versions:
- Version 0.x (legacy): last_node_id/last_link_id, links as arrays or objects
- Version 1 (current): UUID id, state object, links as objects only

Example usage:
    from lib.comfyui_workflow_schema import (
        validate_workflow_file,
        validate_workflow_dict,
        convert_workflow_to_v1,
        SchemaVersion,
    )

    # Validate a file
    report = validate_workflow_file("workflow.json")
    if report.valid:
        print("Workflow is valid!")
    else:
        for error in report.errors:
            print(f"Error: {error.message}")

    # Convert V0 to V1
    v1_workflow = convert_workflow_to_v1(v0_data)
"""

# Types
# CLI
from .cli import main

# Converters
from .converters import (
    convert_links_to_v1,
    convert_workflow_to_v1,
)
from .types import (
    UUID_PATTERN,
    SchemaValidationResult,
    SchemaVersion,
    WorkflowValidationReport,
    generate_uuid,
    is_valid_uuid,
)

# Validator
from .validator import (
    ComfyUIWorkflowValidator,
    validate_workflow_dict,
    validate_workflow_file,
)

__all__ = [
    # Types
    "SchemaVersion",
    "SchemaValidationResult",
    "UUID_PATTERN",
    "WorkflowValidationReport",
    "generate_uuid",
    "is_valid_uuid",
    # Converters
    "convert_links_to_v1",
    "convert_workflow_to_v1",
    # Validator
    "ComfyUIWorkflowValidator",
    "validate_workflow_file",
    "validate_workflow_dict",
    # CLI
    "main",
]
