"""ComfyUI node publishing pipeline.

Public API for packaging, validating, and publishing ComfyUI custom
nodes to the Comfy Registry or ComfyUI Manager ecosystem.

Example usage::

    from comfyui_node_publishing import package_node, validate_package

    config = PackageConfig(
        name="my-custom-node",
        version="1.0.0",
        description="A custom image filter node",
        author="Author Name",
        repository_url="https://github.com/author/my-custom-node",
    )

    pkg_path = package_node("./nodes/my_node.py", config)
    issues = validate_package(pkg_path)
    assert not issues, f"Package has issues: {issues}"
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .registry import RegistryEntry, RegistryMetadataGenerator
from .scaffolder import PackageConfig, PackageScaffolder
from .versioning import VersionError, VersionManager


def package_node(
    node_dir: str,
    config: PackageConfig,
    output_dir: str | None = None,
) -> str:
    """Create a publishable ComfyUI node package.

    Args:
        node_dir: Path to the generated node ``.py`` file.
        config: Package metadata configuration.
        output_dir: Directory to create the package in.
            Defaults to the parent of *node_dir*.

    Returns:
        Absolute path to the created package directory.
    """
    if output_dir is None:
        output_dir = str(Path(node_dir).parent)

    scaffolder = PackageScaffolder()
    return scaffolder.scaffold(node_dir, config, output_dir)


def validate_package(package_dir: str) -> list[str]:
    """Validate a scaffolded package for completeness and correctness.

    Checks directory structure, required files, metadata validity,
    and dependency conflicts.

    Args:
        package_dir: Path to the package directory.

    Returns:
        List of human-readable issues. Empty list means valid.
    """
    issues: list[str] = []
    pkg = Path(package_dir)

    required_files = [
        "__init__.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "LICENSE",
    ]
    for fname in required_files:
        if not (pkg / fname).exists():
            issues.append(f"Missing required file: {fname}")

    init_path = pkg / "__init__.py"
    if init_path.exists():
        content = init_path.read_text(encoding="utf-8")
        if "NODE_CLASS_MAPPINGS" not in content:
            issues.append("__init__.py missing NODE_CLASS_MAPPINGS export")
        if "NODE_DISPLAY_NAME_MAPPINGS" not in content:
            issues.append("__init__.py missing NODE_DISPLAY_NAME_MAPPINGS export")

    pyproject_path = pkg / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        if "[project]" not in content:
            issues.append("pyproject.toml missing [project] section")
        if "[tool.comfy]" not in content:
            issues.append("pyproject.toml missing [tool.comfy] section")

    return issues


def publish_node(
    package_dir: str,
    target: str = "registry",
) -> str:
    """Generate publishing instructions for a packaged node.

    For ``target="registry"``: ensures ``pyproject.toml`` contains
    ``[tool.comfy]`` and returns instructions for ``comfy node publish``.

    For ``target="manager"``: generates a ComfyUI Manager registry
    entry and returns PR instructions.

    Args:
        package_dir: Path to the scaffolded package directory.
        target: Publishing target — ``"registry"`` or ``"manager"``.

    Returns:
        Human-readable instructions for completing publication.

    Raises:
        ValueError: If *target* is not recognized or package is invalid.
    """
    issues = validate_package(package_dir)
    if issues:
        msg = f"Package validation failed: {'; '.join(issues)}"
        raise ValueError(msg)

    if target == "registry":
        return _registry_instructions(package_dir)
    if target == "manager":
        return _manager_instructions(package_dir)

    msg = f"Unknown publish target: {target!r}. Use 'registry' or 'manager'."
    raise ValueError(msg)


def _registry_instructions(package_dir: str) -> str:
    """Generate Comfy Registry publishing instructions.

    Args:
        package_dir: Path to the package directory.

    Returns:
        Markdown-formatted instruction string.
    """
    return textwrap.dedent(f"""\
        ## Publish to Comfy Registry

        1. Install the Comfy CLI:
           ```bash
           pip install comfy-cli
           ```

        2. Log in to your Comfy Registry account:
           ```bash
           comfy node init
           ```

        3. Publish from the package directory:
           ```bash
           cd {package_dir}
           comfy node publish
           ```

        4. Verify the listing at https://registry.comfy.org
    """)


def _manager_instructions(package_dir: str) -> str:
    """Generate ComfyUI Manager PR instructions.

    Args:
        package_dir: Path to the package directory.

    Returns:
        Markdown-formatted instruction string.
    """
    return textwrap.dedent(f"""\
        ## Publish via ComfyUI Manager

        1. Push your node package to a public Git repository.

        2. Fork https://github.com/ltdrdata/ComfyUI-Manager

        3. Add your node entry to ``custom-node-list.json``:
           ```json
           {{
               "reference": "https://github.com/your/repo.git",
               "title": "Your Node Title",
               "description": "Short description",
               "author": "Your Name",
               "install_type": "git-clone",
               "tags": [],
               "pip": []
           }}
           ```

        4. Open a Pull Request to the ComfyUI-Manager repository.

        Package location: {package_dir}
    """)


__all__ = [
    # Public API
    "package_node",
    "publish_node",
    "validate_package",
    # Types
    "PackageConfig",
    "PackageScaffolder",
    "RegistryEntry",
    "RegistryMetadataGenerator",
    "VersionError",
    "VersionManager",
]
