"""Automated test harness for validating generated ComfyUI nodes.

Provides import testing, type checking, mock execution, and widget
validation for generated node code without requiring ComfyUI runtime.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import NodeSpec, WidgetType

# All valid ComfyUI types (both standard and custom)
VALID_COMFYUI_TYPES: set[str] = {t.value for t in WidgetType}


@dataclass
class ValidationResult:
    """Result from node validation.

    Attributes:
        passed: Overall pass/fail status.
        import_ok: Whether the module imported successfully.
        type_check_ok: Whether INPUT_TYPES/RETURN_TYPES are valid.
        execution_ok: Whether mock execution completed without error.
        widget_check_ok: Whether all widget types are valid.
        errors: List of error descriptions.
        warnings: List of warning descriptions.
    """

    passed: bool = True
    import_ok: bool = False
    type_check_ok: bool = False
    execution_ok: bool = False
    widget_check_ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record an error and mark validation as failed.

        Args:
            msg: Error description.
        """
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Record a non-fatal warning.

        Args:
            msg: Warning description.
        """
        self.warnings.append(msg)


def _load_module_from_source(
    source: str, module_name: str = "test_node"
) -> types.ModuleType | None:
    """Load a Python module from source string.

    Args:
        source: Python source code.
        module_name: Name to assign to the module.

    Returns:
        Loaded module or None if import failed.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)
        sys.modules.pop(module_name, None)


def _get_node_class(module: types.ModuleType) -> type | None:
    """Find the node class in a loaded module.

    Looks for NODE_CLASS_MAPPINGS first, then falls back to finding
    any class with INPUT_TYPES classmethod.

    Args:
        module: Loaded Python module.

    Returns:
        The node class, or None if not found.
    """
    # Try NODE_CLASS_MAPPINGS
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if isinstance(mappings, dict) and mappings:
        return next(iter(mappings.values()))

    # Fallback: find class with INPUT_TYPES
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, "INPUT_TYPES"):
            return obj

    return None


def _generate_dummy_value(type_name: str) -> Any:
    """Generate a dummy value for a given ComfyUI type.

    Args:
        type_name: ComfyUI type string (e.g., "INT", "IMAGE").

    Returns:
        A safe dummy value for testing.
    """
    dummy_map: dict[str, Any] = {
        "INT": 1,
        "FLOAT": 1.0,
        "STRING": "test",
        "BOOLEAN": True,
        "COMBO": "option_a",
        "IMAGE": [[[[0.0, 0.0, 0.0]]]],  # Minimal 1x1 RGB
        "MASK": [[[0.0]]],  # Minimal 1x1
        "MODEL": None,
        "CLIP": None,
        "VAE": None,
        "CONDITIONING": [("test", {})],
        "LATENT": {"samples": [[[0.0]]]},
    }
    return dummy_map.get(type_name)


def check_import(source: str) -> tuple[bool, types.ModuleType | None, str]:
    """Test whether generated source can be imported.

    Args:
        source: Python source code.

    Returns:
        Tuple of (success, module_or_none, error_message).
    """
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, None, f"Syntax error: {exc}"

    module = _load_module_from_source(source)
    if module is None:
        return False, None, "Module failed to load"

    return True, module, ""


def check_types(node_class: type) -> tuple[bool, list[str]]:
    """Validate INPUT_TYPES and RETURN_TYPES structure.

    Args:
        node_class: The ComfyUI node class to check.

    Returns:
        Tuple of (valid, list_of_errors).
    """
    errors: list[str] = []

    # Check INPUT_TYPES
    if not hasattr(node_class, "INPUT_TYPES"):
        errors.append("Missing INPUT_TYPES classmethod")
    else:
        try:
            input_types = node_class.INPUT_TYPES()
            if not isinstance(input_types, dict):
                errors.append("INPUT_TYPES must return a dict")
            elif "required" not in input_types:
                errors.append("INPUT_TYPES missing 'required' key")
            else:
                for key in ("required", "optional"):
                    section = input_types.get(key, {})
                    if not isinstance(section, dict):
                        errors.append(f"INPUT_TYPES['{key}'] must be a dict")
                        continue
                    for param_name, param_def in section.items():
                        if not isinstance(param_def, tuple) or len(param_def) < 2:
                            errors.append(
                                f"Input '{param_name}' must be a tuple of (type_str, config_dict)"
                            )
        except Exception as exc:
            errors.append(f"INPUT_TYPES() raised: {exc}")

    # Check RETURN_TYPES
    if not hasattr(node_class, "RETURN_TYPES"):
        errors.append("Missing RETURN_TYPES")
    else:
        rt = node_class.RETURN_TYPES
        if not isinstance(rt, tuple):
            errors.append("RETURN_TYPES must be a tuple")

    return len(errors) == 0, errors


def check_widgets(node_class: type) -> tuple[bool, list[str]]:
    """Validate all widget types are known ComfyUI types.

    Args:
        node_class: The ComfyUI node class to check.

    Returns:
        Tuple of (valid, list_of_errors).
    """
    errors: list[str] = []

    if not hasattr(node_class, "INPUT_TYPES"):
        return False, ["No INPUT_TYPES to check"]

    try:
        input_types = node_class.INPUT_TYPES()
        for section_name in ("required", "optional"):
            section = input_types.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for param_name, param_def in section.items():
                if isinstance(param_def, tuple) and len(param_def) >= 1:
                    type_str = param_def[0]
                    if type_str not in VALID_COMFYUI_TYPES:
                        errors.append(
                            f"Unknown widget type '{type_str}' for input '{param_name}'"
                        )
    except Exception as exc:
        errors.append(f"Error checking widgets: {exc}")

    return len(errors) == 0, errors


def check_execution(node_class: type, spec: NodeSpec | None = None) -> tuple[bool, str]:
    """Mock-execute the node's FUNCTION with dummy inputs.

    Args:
        node_class: The ComfyUI node class to check.
        spec: Optional NodeSpec for better dummy generation.

    Returns:
        Tuple of (success, error_message).
    """
    func_name = getattr(node_class, "FUNCTION", "execute")
    func = getattr(node_class, func_name, None)
    if func is None:
        return False, f"Missing function '{func_name}'"

    # Build dummy kwargs from INPUT_TYPES
    kwargs: dict[str, Any] = {}
    try:
        input_types = node_class.INPUT_TYPES()
        for section in ("required", "optional"):
            for param_name, param_def in input_types.get(section, {}).items():
                if isinstance(param_def, tuple) and len(param_def) >= 1:
                    kwargs[param_name] = _generate_dummy_value(param_def[0])
    except Exception:
        pass

    try:
        result = func(**kwargs)
        # Verify result is a tuple matching RETURN_TYPES length
        return_types = getattr(node_class, "RETURN_TYPES", ())
        if isinstance(result, tuple) and len(result) != len(return_types):
            return False, (
                f"Return tuple length {len(result)} != RETURN_TYPES length {len(return_types)}"
            )
        return True, ""
    except Exception as exc:
        return False, f"Execution error: {exc}"


def validate_node(source: str, spec: NodeSpec | None = None) -> ValidationResult:
    """Run full validation suite on generated node source.

    Performs import, type, widget, and execution checks.

    Args:
        source: Generated Python source code.
        spec: Optional NodeSpec for enhanced validation.

    Returns:
        ValidationResult with all check outcomes.
    """
    result = ValidationResult()

    # 1. Import check
    import_ok, module, import_err = check_import(source)
    result.import_ok = import_ok
    if not import_ok:
        result.add_error(f"Import failed: {import_err}")
        return result

    # 2. Find node class
    node_class = _get_node_class(module)
    if node_class is None:
        result.add_error("No node class found in generated code")
        return result

    # 3. Type check
    type_ok, type_errors = check_types(node_class)
    result.type_check_ok = type_ok
    for err in type_errors:
        result.add_error(f"Type check: {err}")

    # 4. Widget check
    widget_ok, widget_errors = check_widgets(node_class)
    result.widget_check_ok = widget_ok
    for err in widget_errors:
        result.add_error(f"Widget check: {err}")

    # 5. Execution check
    exec_ok, exec_err = check_execution(node_class, spec)
    result.execution_ok = exec_ok
    if not exec_ok:
        result.add_error(f"Execution: {exec_err}")

    return result
