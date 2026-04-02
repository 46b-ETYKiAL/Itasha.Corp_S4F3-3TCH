"""Code generator for ComfyUI custom nodes.

Produces valid ComfyUI node Python files from NodeSpec objects,
supporting both V3 (stateless classmethod) and V1 (legacy) formats.
All generated code is AST-validated and security-checked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from jinja2 import BaseLoader, Environment

from .types import InputSpec, NodeSpec

# --- Security ---

_BLOCKED_PATTERNS: set[str] = {
    "os.system",
    "subprocess",
    "eval",
    "exec",
    "__import__",
}


def _check_ast_security(source: str) -> list[str]:
    """Scan AST for blocked patterns.

    Args:
        source: Python source code to check.

    Returns:
        List of security violation descriptions. Empty if clean.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        violations.append(f"SyntaxError: {exc}")
        return violations

    for node in ast.walk(tree):
        # Check function calls
        if isinstance(node, ast.Call):
            func = node.func
            # os.system(...)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                full_name = f"{func.value.id}.{func.attr}"
                if full_name in _BLOCKED_PATTERNS:
                    violations.append(f"Blocked call: {full_name}")
            # eval(...), exec(...)
            if isinstance(func, ast.Name) and func.id in _BLOCKED_PATTERNS:
                violations.append(f"Blocked builtin: {func.id}")
        # Check imports: import subprocess, from subprocess import ...
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BLOCKED_PATTERNS:
                    violations.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _BLOCKED_PATTERNS:
                violations.append(f"Blocked import from: {node.module}")

    return violations


# --- Templates ---

_V3_TEMPLATE = """\
\"\"\"ComfyUI custom node: {{ spec.display_name }}.

{{ spec.description }}
\"\"\"
from __future__ import annotations

from typing import Any


class {{ class_name }}:
    \"\"\"{{ spec.display_name }}.

    {{ spec.description }}
    \"\"\"

    CATEGORY = "{{ spec.category }}"
    FUNCTION = "{{ spec.function_name }}"
    {% if spec.is_output_node %}OUTPUT_NODE = True{% endif %}

    RETURN_TYPES = ({{ return_types }})
    RETURN_NAMES = ({{ return_names }})

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        \"\"\"Define node input types.\"\"\"
        return {
            "required": {
{% for inp in required_inputs %}
                "{{ inp.name }}": (
                    "{{ inp.widget.widget_type.value }}",
                    {{ inp_config(inp) }},
                ),
{% endfor %}
            },
{% if optional_inputs %}
            "optional": {
{% for inp in optional_inputs %}
                "{{ inp.name }}": (
                    "{{ inp.widget.widget_type.value }}",
                    {{ inp_config(inp) }},
                ),
{% endfor %}
            },
{% endif %}
        }

    @classmethod
    def {{ spec.function_name }}(
        cls,
{% for inp in spec.inputs %}
        {{ inp.name }}: Any = None,
{% endfor %}
    ) -> tuple[Any, ...]:
        \"\"\"Execute the node.

        Args:
{% for inp in spec.inputs %}
            {{ inp.name }}: {{ inp.tooltip or inp.name }}.
{% endfor %}

        Returns:
            Tuple of ({{ return_names_plain }}).
        \"\"\"
        # TODO: Implement node logic here
{% for out in spec.outputs %}
        {{ out.name }}_result = None  # placeholder for {{ out.name }}
{% endfor %}
        return ({{ result_tuple }})
"""

_V1_SHIM_TEMPLATE = """\

# V1 compatibility mappings
NODE_CLASS_MAPPINGS = {
    "{{ spec.name }}": {{ class_name }},
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "{{ spec.name }}": "{{ spec.display_name }}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
"""

_PYPROJECT_TEMPLATE = """\
[project]
name = "comfyui-{{ spec.name | lower | replace('_', '-') }}"
version = "0.1.0"
description = "{{ spec.description or spec.display_name }}"
requires-python = ">=3.10"

[tool.comfy]
PublisherId = ""
DisplayName = "{{ spec.display_name }}"
Icon = ""

[tool.comfy.custom_nodes.{{ spec.name }}]
category = "{{ spec.category }}"
is_output_node = {{ spec.is_output_node | lower }}
"""


def _build_input_config(inp: InputSpec) -> str:
    """Build the config dict string for an input.

    Args:
        inp: Input specification.

    Returns:
        Python dict literal string for the input config.
    """
    parts: list[str] = []
    cfg = inp.widget
    if cfg.default is not None:
        parts.append(f'"default": {cfg.default!r}')
    if cfg.min_value is not None:
        parts.append(f'"min": {cfg.min_value}')
    if cfg.max_value is not None:
        parts.append(f'"max": {cfg.max_value}')
    if cfg.step is not None:
        parts.append(f'"step": {cfg.step}')
    if cfg.multiline:
        parts.append('"multiline": True')
    if cfg.choices:
        parts.append(f'"choices": {cfg.choices!r}')
    if inp.tooltip:
        parts.append(f'"tooltip": {inp.tooltip!r}')
    if not parts:
        return "{}"
    return "{" + ", ".join(parts) + "}"


@dataclass
class GenerationResult:
    """Result from code generation.

    Attributes:
        source: Generated Python source code.
        pyproject: Generated pyproject.toml content.
        ast_valid: Whether the source passed AST validation.
        security_violations: List of security check failures.
        file_path: Path where the file was written, if any.
    """

    source: str
    pyproject: str
    ast_valid: bool
    security_violations: list[str]
    file_path: Path | None = None


def _to_class_name(name: str) -> str:
    """Convert a node name to a PascalCase class name.

    Args:
        name: Snake_case or arbitrary node name.

    Returns:
        PascalCase class name.
    """
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def generate_node_code(spec: NodeSpec) -> GenerationResult:
    """Generate ComfyUI node Python source from a NodeSpec.

    Produces V3-format code by default with optional V1 compatibility
    shim appended. All generated code is AST-validated and
    security-checked.

    Args:
        spec: Complete node specification.

    Returns:
        GenerationResult with source, pyproject, and validation info.
    """
    class_name = _to_class_name(spec.name)
    required_inputs = [i for i in spec.inputs if i.required]
    optional_inputs = [i for i in spec.inputs if not i.required]

    # Build return type/name tuples
    if spec.outputs:
        return_types = ", ".join(f'"{o.type.value}"' for o in spec.outputs)
        return_names = ", ".join(f'"{o.name}"' for o in spec.outputs)
        return_names_plain = ", ".join(o.name for o in spec.outputs)
        result_tuple = ", ".join(f"{o.name}_result" for o in spec.outputs)
    else:
        return_types = ""
        return_names = ""
        return_names_plain = ""
        result_tuple = ""

    # Add trailing comma for single-element tuples
    if len(spec.outputs) == 1:
        return_types += ","
        return_names += ","
        result_tuple += ","

    # autoescape=False is intentional: we generate Python source code, not HTML.
    # Enabling autoescape would corrupt the output by escaping valid Python syntax.
    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 — generating Python source, not HTML
    )
    env.globals["inp_config"] = _build_input_config

    # Render V3 template
    template = env.from_string(_V3_TEMPLATE)
    source = template.render(
        spec=spec,
        class_name=class_name,
        required_inputs=required_inputs,
        optional_inputs=optional_inputs,
        return_types=return_types,
        return_names=return_names,
        return_names_plain=return_names_plain,
        result_tuple=result_tuple,
    )

    # Append V1 shim if not pure V3
    if not spec.v3_format:
        shim_template = env.from_string(_V1_SHIM_TEMPLATE)
        source += shim_template.render(spec=spec, class_name=class_name)

    # Always append V1 shim for backward compatibility
    if spec.v3_format:
        shim_template = env.from_string(_V1_SHIM_TEMPLATE)
        source += shim_template.render(spec=spec, class_name=class_name)

    # AST validation
    ast_valid = True
    try:
        tree = ast.parse(source)
        compile(tree, "<generated>", "exec")
    except SyntaxError:
        ast_valid = False

    # Security check
    security_violations = _check_ast_security(source)

    # Generate pyproject.toml
    pyproject_template = env.from_string(_PYPROJECT_TEMPLATE)
    pyproject = pyproject_template.render(spec=spec)

    return GenerationResult(
        source=source,
        pyproject=pyproject,
        ast_valid=ast_valid,
        security_violations=security_violations,
    )


def write_node_package(
    spec: NodeSpec,
    output_dir: Path,
) -> GenerationResult:
    """Generate and write a complete node package to disk.

    Creates the output directory with the generated Python file
    and pyproject.toml.

    Args:
        spec: Complete node specification.
        output_dir: Directory to write the package into.

    Returns:
        GenerationResult with file_path set.
    """
    result = generate_node_code(spec)

    output_dir.mkdir(parents=True, exist_ok=True)

    node_file = output_dir / f"{spec.name}.py"
    node_file.write_text(result.source, encoding="utf-8")

    pyproject_file = output_dir / "pyproject.toml"
    pyproject_file.write_text(result.pyproject, encoding="utf-8")

    result.file_path = node_file
    return result
