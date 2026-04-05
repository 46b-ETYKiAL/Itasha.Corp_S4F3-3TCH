"""Workflow template manager for ComfyUI.

Manages workflow templates with categories, variable substitution,
import from ComfyUI JSON exports, and export for sharing.

Templates are stored as JSON files in a configurable directory with
metadata including category, description, variables, and tags.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(
    {
        "txt2img",
        "img2img",
        "inpainting",
        "upscaling",
        "controlnet",
        "ip_adapter",
        "batch",
    }
)

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")

_ABSOLUTE_PATH_PATTERN = re.compile(
    r'(?:[A-Z]:\\\\|[A-Z]:/|/home/|/Users/|/root/)[^\s"\']*',
    re.IGNORECASE,
)


@dataclasses.dataclass
class WorkflowTemplate:
    """A reusable ComfyUI workflow template.

    Attributes:
        name: Unique template identifier.
        category: Workflow category (must be in VALID_CATEGORIES).
        description: Human-readable description.
        workflow: The ComfyUI workflow JSON structure.
        variables: Mapping of variable names to default values.
        version: Template schema version.
        tags: Optional tags for filtering.
    """

    name: str
    category: str
    description: str
    workflow: dict[str, Any]
    variables: dict[str, Any]
    version: str = "1.0"
    tags: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate template fields.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        if not self.name or not self.name.strip():
            errors.append("Template name must not be empty.")
        if self.category not in VALID_CATEGORIES:
            errors.append(
                f"Invalid category '{self.category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )
        if not isinstance(self.workflow, dict) or not self.workflow:
            errors.append("Workflow must be a non-empty dict.")
        return errors


class TemplateManager:
    """Manages ComfyUI workflow templates on disk.

    Templates are persisted as individual JSON files in the configured
    template directory. Each file contains the full template metadata
    alongside the workflow definition.

    Args:
        template_dir: Directory path for storing template JSON files.
    """

    def __init__(self, template_dir: str | Path) -> None:
        self._dir = Path(template_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def template_dir(self) -> Path:
        """Return the template storage directory."""
        return self._dir

    def list_templates(self, category: str | None = None) -> list[WorkflowTemplate]:
        """List all templates, optionally filtered by category.

        Args:
            category: If provided, only return templates in this category.

        Returns:
            List of matching WorkflowTemplate objects.
        """
        templates: list[WorkflowTemplate] = []
        for path in sorted(self._dir.glob("*.json")):
            tpl = self._load_from_file(path)
            if tpl is None:
                continue
            if category is not None and tpl.category != category:
                continue
            templates.append(tpl)
        return templates

    def get_template(self, name: str) -> WorkflowTemplate | None:
        """Retrieve a template by name.

        Args:
            name: Template name to look up.

        Returns:
            The matching template, or None if not found.
        """
        path = self._template_path(name)
        if not path.exists():
            return None
        return self._load_from_file(path)

    def import_workflow(
        self,
        json_path: str,
        name: str,
        category: str,
        *,
        description: str = "",
    ) -> WorkflowTemplate:
        """Import a ComfyUI workflow JSON export as a template.

        Loads the workflow, detects template variables, strips absolute
        paths, and saves as a managed template.

        Args:
            json_path: Path to the ComfyUI JSON export file.
            name: Name for the new template.
            category: Category to assign.
            description: Optional description.

        Returns:
            The created WorkflowTemplate.

        Raises:
            FileNotFoundError: If json_path does not exist.
            ValueError: If the JSON is not a valid workflow.
        """
        source = Path(json_path)
        if not source.exists():
            msg = f"Workflow file not found: {json_path}"
            raise FileNotFoundError(msg)

        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = "Workflow JSON must be a dict at top level."
            raise ValueError(msg)

        workflow = self._strip_absolute_paths(raw)
        variables = self._detect_variables(workflow)

        template = WorkflowTemplate(
            name=name,
            category=category,
            description=description or f"Imported from {source.name}",
            workflow=workflow,
            variables=variables,
        )

        errors = template.validate()
        if errors:
            msg = f"Template validation failed: {'; '.join(errors)}"
            raise ValueError(msg)

        self.save_template(template)
        logger.info("Imported workflow '%s' from %s", name, json_path)
        return template

    def export_template(self, name: str, output_path: str) -> str:
        """Export a template for sharing.

        Strips absolute paths and writes a clean JSON file suitable
        for distribution.

        Args:
            name: Template name to export.
            output_path: Destination file path.

        Returns:
            The resolved output path.

        Raises:
            KeyError: If the template does not exist.
        """
        template = self.get_template(name)
        if template is None:
            msg = f"Template '{name}' not found."
            raise KeyError(msg)

        export_data = self._template_to_dict(template)
        export_data["workflow"] = self._strip_absolute_paths(
            copy.deepcopy(template.workflow)
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported template '%s' to %s", name, output_path)
        return str(out)

    def render_template(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
    ) -> dict:
        """Render a template by substituting variables.

        Variables in the workflow are encoded as ``{{variable_name}}``.
        Provided values override defaults; missing variables use their
        default values from the template definition.

        Args:
            name: Template name.
            variables: Variable overrides (name → value).

        Returns:
            The rendered workflow dict.

        Raises:
            KeyError: If the template does not exist.
        """
        template = self.get_template(name)
        if template is None:
            msg = f"Template '{name}' not found."
            raise KeyError(msg)

        merged = {**template.variables, **(variables or {})}
        rendered = copy.deepcopy(template.workflow)
        return self._substitute_variables(rendered, merged)

    def save_template(self, template: WorkflowTemplate) -> None:
        """Persist a template to disk.

        Args:
            template: The WorkflowTemplate to save.

        Raises:
            ValueError: If the template fails validation.
        """
        errors = template.validate()
        if errors:
            msg = f"Cannot save invalid template: {'; '.join(errors)}"
            raise ValueError(msg)

        path = self._template_path(template.name)
        data = self._template_to_dict(template)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Saved template '%s' to %s", template.name, path)

    def delete_template(self, name: str) -> bool:
        """Delete a template from disk.

        Args:
            name: Template name to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._template_path(name)
        if path.exists():
            path.unlink()
            logger.info("Deleted template '%s'", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _template_path(self, name: str) -> Path:
        """Build the file path for a named template."""
        safe_name = re.sub(r"[^\w\-]", "_", name)
        return self._dir / f"{safe_name}.json"

    def _load_from_file(self, path: Path) -> WorkflowTemplate | None:
        """Load a WorkflowTemplate from a JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowTemplate(
                name=data["name"],
                category=data["category"],
                description=data.get("description", ""),
                workflow=data["workflow"],
                variables=data.get("variables", {}),
                version=data.get("version", "1.0"),
                tags=data.get("tags", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to load template from %s", path)
            return None

    @staticmethod
    def _template_to_dict(template: WorkflowTemplate) -> dict[str, Any]:
        """Convert a WorkflowTemplate to a serialisable dict."""
        return {
            "name": template.name,
            "category": template.category,
            "description": template.description,
            "workflow": template.workflow,
            "variables": template.variables,
            "version": template.version,
            "tags": template.tags,
        }

    @staticmethod
    def _strip_absolute_paths(obj: Any) -> Any:
        """Recursively replace absolute file paths with basenames."""
        if isinstance(obj, str):
            return _ABSOLUTE_PATH_PATTERN.sub(lambda m: Path(m.group()).name, obj)
        if isinstance(obj, dict):
            return {k: TemplateManager._strip_absolute_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TemplateManager._strip_absolute_paths(v) for v in obj]
        return obj

    @staticmethod
    def _detect_variables(workflow: dict[str, Any]) -> dict[str, Any]:
        """Scan workflow for ``{{var}}`` placeholders and build defaults."""
        variables: dict[str, Any] = {}
        raw = json.dumps(workflow)
        for match in _VARIABLE_PATTERN.finditer(raw):
            var_name = match.group(1)
            if var_name not in variables:
                variables[var_name] = ""
        return variables

    @staticmethod
    def _substitute_variables(obj: Any, variables: dict[str, Any]) -> Any:
        """Recursively substitute ``{{var}}`` in strings."""
        if isinstance(obj, str):

            def _replacer(m: re.Match) -> str:
                key = m.group(1)
                val = variables.get(key, m.group(0))
                return str(val)

            return _VARIABLE_PATTERN.sub(_replacer, obj)
        if isinstance(obj, dict):
            return {
                k: TemplateManager._substitute_variables(v, variables)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [TemplateManager._substitute_variables(v, variables) for v in obj]
        return obj
