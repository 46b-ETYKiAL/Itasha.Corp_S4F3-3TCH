"""
ComfyUI Workflow Validator - Common Validations.

Contains validation logic shared between V0 and V1 schemas:
- Node validation
- Node-link consistency checking
- Helper functions for building lookups
"""

from typing import TYPE_CHECKING, Any

from .types import SchemaValidationResult

if TYPE_CHECKING:
    from .types import WorkflowValidationReport


def validate_nodes(
    data: dict[str, Any], report: "WorkflowValidationReport", version: int
) -> None:
    """Validate the nodes array."""
    nodes = data.get("nodes")

    if nodes is None:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="nodes",
                message="Missing required 'nodes' field",
                severity="error",
            )
        )
        return

    if isinstance(nodes, dict):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="nodes",
                message="'nodes' must be an array, not an object",
                severity="error",
                details={"fix": "Convert nodes from object to array format"},
            )
        )
        return

    if not isinstance(nodes, list):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field="nodes",
                message=f"'nodes' must be an array, got: {type(nodes).__name__}",
                severity="error",
            )
        )
        return

    # Validate each node
    node_ids: set[Any] = set()
    for i, node in enumerate(nodes):
        validate_single_node(node, i, node_ids, report)


def validate_single_node(
    node: Any, index: int, node_ids: set[Any], report: "WorkflowValidationReport"
) -> None:
    """Validate a single node object."""
    if not isinstance(node, dict):
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field=f"nodes[{index}]",
                message=f"Node must be an object, got: {type(node).__name__}",
                severity="error",
            )
        )
        return

    # Check required fields
    required_fields = ["id", "type"]
    for field_name in required_fields:
        if field_name not in node:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"nodes[{index}].{field_name}",
                    message=f"Node missing required field: {field_name}",
                    severity="error",
                )
            )

    # Check for duplicate IDs
    node_id = node.get("id")
    if node_id is not None:
        if node_id in node_ids:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"nodes[{index}].id",
                    message=f"Duplicate node ID: {node_id}",
                    severity="error",
                )
            )
        node_ids.add(node_id)

    # Check recommended fields
    recommended = ["pos", "size", "inputs", "outputs"]
    missing = [f for f in recommended if f not in node]
    if missing:
        report.add_result(
            SchemaValidationResult(
                valid=True,
                field=f"nodes[{index}]",
                message=f"Node missing recommended fields: {missing}",
                severity="warning",
            )
        )


def build_nodes_by_id(nodes: list[Any]) -> dict[Any, dict[str, Any]]:
    """Build node lookup by ID."""
    nodes_by_id: dict[Any, dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, dict) and "id" in node:
            nodes_by_id[node["id"]] = node
    return nodes_by_id


def build_link_info(links: list[Any]) -> dict[Any, dict[str, Any]]:
    """Build link lookup from links list."""
    link_info: dict[Any, dict[str, Any]] = {}
    for link in links:
        if isinstance(link, list) and len(link) >= 5:
            link_info[link[0]] = {
                "origin_id": link[1],
                "origin_slot": link[2],
                "target_id": link[3],
                "target_slot": link[4],
            }
        elif isinstance(link, dict) and link.get("id") is not None:
            link_info[link["id"]] = {
                "origin_id": link.get("origin_id"),
                "origin_slot": link.get("origin_slot"),
                "target_id": link.get("target_id"),
                "target_slot": link.get("target_slot"),
            }
    return link_info


def validate_link_node_references(
    link_info: dict[Any, dict[str, Any]],
    nodes_by_id: dict[Any, dict[str, Any]],
    report: "WorkflowValidationReport",
) -> None:
    """Validate links reference valid nodes."""
    for link_id, info in link_info.items():
        if info["origin_id"] not in nodes_by_id:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"link[{link_id}]",
                    message=f"Link references non-existent source node: {info['origin_id']}",
                    severity="error",
                )
            )
        if info["target_id"] not in nodes_by_id:
            report.add_result(
                SchemaValidationResult(
                    valid=False,
                    field=f"link[{link_id}]",
                    message=f"Link references non-existent target node: {info['target_id']}",
                    severity="error",
                )
            )


def validate_output_slot_links(
    link_id: int,
    origin_id: int,
    origin_slot: int,
    nodes_by_id: dict[Any, dict[str, Any]],
    report: "WorkflowValidationReport",
) -> None:
    """Validate source node output slot contains link ID."""
    if origin_id not in nodes_by_id:
        return
    outputs = nodes_by_id[origin_id].get("outputs", [])
    if (
        not isinstance(outputs, list)
        or origin_slot is None
        or origin_slot >= len(outputs)
    ):
        return
    output = outputs[origin_slot]
    if not isinstance(output, dict):
        return
    output_links = output.get("links", [])
    if isinstance(output_links, list) and link_id not in output_links:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field=f"node[{origin_id}].outputs[{origin_slot}].links",
                message=f"Output slot missing link {link_id} in links array",
                severity="error",
                details={
                    "link_id": link_id,
                    "node_id": origin_id,
                    "slot": origin_slot,
                    "current_links": output_links,
                    "fix": f"Add {link_id} to outputs[{origin_slot}].links array",
                },
            )
        )


def validate_input_slot_link(
    link_id: int,
    target_id: int,
    target_slot: int,
    nodes_by_id: dict[Any, dict[str, Any]],
    report: "WorkflowValidationReport",
) -> None:
    """Validate target node input slot references link ID."""
    if target_id not in nodes_by_id:
        return
    inputs = nodes_by_id[target_id].get("inputs", [])
    if (
        not isinstance(inputs, list)
        or target_slot is None
        or target_slot >= len(inputs)
    ):
        return
    inp = inputs[target_slot]
    if not isinstance(inp, dict):
        return
    input_link = inp.get("link")
    if input_link != link_id:
        report.add_result(
            SchemaValidationResult(
                valid=False,
                field=f"node[{target_id}].inputs[{target_slot}].link",
                message=f"Input slot has link={input_link} but should be {link_id}",
                severity="error",
                details={
                    "expected_link": link_id,
                    "actual_link": input_link,
                    "node_id": target_id,
                    "slot": target_slot,
                    "fix": f"Set inputs[{target_slot}].link to {link_id}",
                },
            )
        )


def validate_node_link_consistency(
    data: dict[str, Any], report: "WorkflowValidationReport", version: int
) -> None:
    """Validate that links reference valid node IDs and node outputs/inputs reference valid links."""
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    if not isinstance(nodes, list) or not isinstance(links, list):
        return

    nodes_by_id = build_nodes_by_id(nodes)
    link_info = build_link_info(links)

    validate_link_node_references(link_info, nodes_by_id, report)

    for link_id, info in link_info.items():
        validate_output_slot_links(
            link_id, info["origin_id"], info["origin_slot"], nodes_by_id, report
        )
        validate_input_slot_link(
            link_id, info["target_id"], info["target_slot"], nodes_by_id, report
        )
