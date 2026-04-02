"""
ComfyUI Workflow Converters

Utilities for converting between workflow schema versions.
"""

from typing import Any

from .types import generate_uuid, is_valid_uuid


def convert_links_to_v1(links: list[Any]) -> list[dict[str, Any]]:
    """
    Convert V0 array-style links to V1 object-style links.

    Args:
        links: List of links (arrays or objects)

    Returns:
        List of link objects in V1 format
    """
    result = []
    for link in links:
        if isinstance(link, list) and len(link) >= 6:
            result.append(
                {
                    "id": link[0],
                    "origin_id": link[1],
                    "origin_slot": link[2],
                    "target_id": link[3],
                    "target_slot": link[4],
                    "type": link[5],
                }
            )
        elif isinstance(link, dict):
            result.append(link)
    return result


def convert_workflow_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a V0 workflow to V1 format.

    Args:
        data: V0 workflow dictionary

    Returns:
        V1 workflow dictionary
    """
    result = dict(data)

    # Set version to 1
    result["version"] = 1

    # Add UUID id if not present or not valid
    if "id" not in result or not is_valid_uuid(str(result.get("id", ""))):
        result["id"] = generate_uuid()

    # Convert last_* fields to state object
    state = result.get("state", {})
    if not isinstance(state, dict):
        state = {}

    state.setdefault("lastNodeId", result.pop("last_node_id", 0))
    state.setdefault("lastLinkId", result.pop("last_link_id", 0))
    state.setdefault("lastGroupId", 0)
    state.setdefault("lastRerouteId", 0)
    result["state"] = state

    # Remove legacy fields
    result.pop("last_node_id", None)
    result.pop("last_link_id", None)

    # Convert links to object format
    if "links" in result:
        result["links"] = convert_links_to_v1(result["links"])

    return result
