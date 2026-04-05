"""Multi-output composite node support for ComfyUI.

Extends the generator to handle nodes with multiple outputs,
correctly generating RETURN_TYPES, RETURN_NAMES, and tuple
return handling in the FUNCTION method.
"""

from __future__ import annotations

from .generator import GenerationResult, generate_node_code
from .types import NodeSpec, OutputSpec, WidgetType


def validate_composite_outputs(spec: NodeSpec) -> list[str]:
    """Validate that a composite node's outputs are well-formed.

    Args:
        spec: Node specification to validate.

    Returns:
        List of validation errors. Empty if valid.
    """
    errors: list[str] = []

    if len(spec.outputs) < 2:
        errors.append(f"Composite nodes need 2+ outputs, got {len(spec.outputs)}")

    # Check for duplicate output names
    names = [o.name for o in spec.outputs]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            errors.append(f"Duplicate output name: '{name}'")
        seen.add(name)

    return errors


def create_composite_spec(
    name: str,
    category: str,
    description: str,
    outputs: list[tuple[str, WidgetType]],
    **kwargs: object,
) -> NodeSpec:
    """Create a NodeSpec pre-configured for composite (multi-output) use.

    Convenience factory that builds OutputSpec objects from simple
    tuples and validates the composite structure.

    Args:
        name: Internal node name.
        category: Node category path.
        description: Node description.
        outputs: List of (name, type) tuples for each output.
        **kwargs: Additional NodeSpec fields.

    Returns:
        Validated NodeSpec with multiple outputs.

    Raises:
        ValueError: If outputs are invalid for composite use.
    """
    output_specs = [
        OutputSpec(name=out_name, type=out_type) for out_name, out_type in outputs
    ]

    spec = NodeSpec(
        name=name,
        category=category,
        description=description,
        outputs=output_specs,
        **kwargs,  # type: ignore[arg-type]
    )

    errors = validate_composite_outputs(spec)
    if errors:
        raise ValueError(f"Invalid composite spec: {'; '.join(errors)}")

    return spec


def generate_composite_node(spec: NodeSpec) -> GenerationResult:
    """Generate code for a multi-output composite node.

    Validates the composite output structure, then delegates to
    the standard generator which already handles multiple outputs.

    Args:
        spec: Node specification with 2+ outputs.

    Returns:
        GenerationResult with generated source.

    Raises:
        ValueError: If the spec doesn't qualify as composite.
    """
    errors = validate_composite_outputs(spec)
    if errors:
        raise ValueError(f"Invalid composite spec: {'; '.join(errors)}")

    return generate_node_code(spec)


def split_node_spec(
    spec: NodeSpec,
    output_groups: list[list[int]],
) -> list[NodeSpec]:
    """Split a composite node into multiple simpler nodes.

    Useful when a node has too many outputs and should be decomposed
    into a chain of nodes.

    Args:
        spec: Original composite node specification.
        output_groups: List of groups, each containing output indices.

    Returns:
        List of NodeSpec objects, one per group.

    Raises:
        ValueError: If output indices are invalid.
    """
    max_idx = len(spec.outputs) - 1
    all_indices: set[int] = set()
    for group in output_groups:
        for idx in group:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Output index {idx} out of range [0, {max_idx}]")
            if idx in all_indices:
                raise ValueError(f"Duplicate output index: {idx}")
            all_indices.add(idx)

    result: list[NodeSpec] = []
    for i, group in enumerate(output_groups):
        group_outputs = [spec.outputs[idx] for idx in group]
        suffix = f"_part{i + 1}" if len(output_groups) > 1 else ""
        split_spec = spec.model_copy(
            update={
                "name": f"{spec.name}{suffix}",
                "display_name": f"{spec.display_name} (Part {i + 1})",
                "outputs": group_outputs,
            }
        )
        result.append(split_spec)

    return result
