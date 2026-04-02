"""ComfyUI Manager registry entry generator and validator.

Produces JSON entries compatible with both the ComfyUI Manager
``custom-node-list.json`` format and the newer Comfy Registry
format used by ``comfy node publish``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from .scaffolder import PackageConfig

# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

_VALID_INSTALL_TYPES = frozenset({"git-clone", "copy", "pip"})

_GIT_URL_PATTERN = re.compile(
    r"^https?://[a-zA-Z0-9._\-]+\.[a-zA-Z]{2,}/[^\s]+\.git$"
    r"|^https?://github\.com/[^\s]+$"
)


@dataclasses.dataclass
class RegistryEntry:
    """ComfyUI Manager ``custom-node-list.json`` entry.

    Attributes:
        reference: Git URL for the node repository.
        title: Human-readable title shown in Manager.
        description: Short description of the node.
        author: Author name.
        install_type: One of ``git-clone``, ``copy``, ``pip``.
        tags: Discovery tags.
        pip: Additional pip requirements.
    """

    reference: str
    title: str
    description: str
    author: str
    install_type: str = "git-clone"
    tags: list[str] = dataclasses.field(default_factory=list)
    pip: list[str] = dataclasses.field(default_factory=list)


class RegistryMetadataGenerator:
    """Generate and validate ComfyUI registry metadata.

    Supports both the legacy ComfyUI Manager JSON format and the newer
    Comfy Registry format backed by ``pyproject.toml [tool.comfy]``.
    """

    def generate_manager_entry(self, config: PackageConfig) -> RegistryEntry:
        """Create a ComfyUI Manager registry entry from package config.

        Args:
            config: Validated package configuration.

        Returns:
            Populated ``RegistryEntry`` instance.
        """
        reference = config.repository_url
        if reference and not reference.endswith(".git"):
            reference = reference + ".git"

        return RegistryEntry(
            reference=reference,
            title=config.name,
            description=config.description,
            author=config.author,
            install_type="git-clone",
            tags=list(config.tags),
            pip=list(config.dependencies),
        )

    def validate_entry(self, entry: RegistryEntry) -> list[str]:
        """Validate a registry entry for completeness and format.

        Args:
            entry: Registry entry to validate.

        Returns:
            List of human-readable validation error strings.
            Empty list means the entry is valid.
        """
        errors: list[str] = []

        if not entry.reference:
            errors.append("reference (Git URL) is required")
        elif not _GIT_URL_PATTERN.match(entry.reference):
            errors.append(f"reference must be a valid Git URL, got: {entry.reference}")

        if not entry.title:
            errors.append("title is required")
        elif len(entry.title) > 100:
            errors.append("title must be 100 characters or fewer")

        if not entry.description:
            errors.append("description is required")
        elif len(entry.description) > 500:
            errors.append("description must be 500 characters or fewer")

        if not entry.author:
            errors.append("author is required")

        if entry.install_type not in _VALID_INSTALL_TYPES:
            errors.append(f"install_type must be one of {sorted(_VALID_INSTALL_TYPES)}, got: {entry.install_type}")

        for tag in entry.tags:
            if not tag or len(tag) > 50:
                errors.append(f"tag must be non-empty and <=50 chars: {tag!r}")

        for dep in entry.pip:
            if not dep or not re.match(r"^[a-zA-Z0-9_\-.\[\]]+", dep):
                errors.append(f"invalid pip dependency format: {dep!r}")

        return errors

    def to_json(self, entry: RegistryEntry, *, indent: int = 2) -> str:
        """Serialize a registry entry to JSON.

        Args:
            entry: Registry entry to serialize.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self._entry_to_dict(entry)
        return json.dumps(data, indent=indent, sort_keys=False)

    def generate_comfy_registry_metadata(
        self,
        config: PackageConfig,
    ) -> dict[str, Any]:
        """Generate metadata dict for the Comfy Registry ``[tool.comfy]`` format.

        This metadata is embedded in ``pyproject.toml`` and used by
        ``comfy node publish``.

        Args:
            config: Package configuration.

        Returns:
            Dictionary suitable for the ``[tool.comfy]`` TOML section.
        """
        return {
            "PublisherId": "",
            "DisplayName": config.name,
            "Icon": "",
            "Tags": list(config.tags),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_dict(entry: RegistryEntry) -> dict[str, Any]:
        """Convert a registry entry to a plain dictionary.

        Args:
            entry: Registry entry.

        Returns:
            Ordered dictionary matching ``custom-node-list.json`` shape.
        """
        return {
            "reference": entry.reference,
            "title": entry.title,
            "description": entry.description,
            "author": entry.author,
            "install_type": entry.install_type,
            "tags": entry.tags,
            "pip": entry.pip,
        }
