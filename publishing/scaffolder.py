"""Package scaffolder for publishable ComfyUI custom node packages.

Generates a complete directory structure suitable for distribution via
the Comfy Registry (``comfy node publish``) or ComfyUI Manager git-clone
installation.  Produces ``__init__.py``, ``pyproject.toml``,
``requirements.txt``, ``README.md``, ``LICENSE``, and ``.gitignore``.
"""

from __future__ import annotations

import dataclasses
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class PackageConfig:
    """Configuration for a publishable ComfyUI node package.

    Attributes:
        name: Package name (used as directory name and pyproject name).
        version: SemVer version string.
        description: Short human-readable description.
        author: Author name or "Name <email>".
        license: SPDX license identifier.
        repository_url: Git repository URL.
        tags: Discovery tags for registry search.
        dependencies: Python package requirements (pip format).
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    repository_url: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)
    dependencies: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Licence templates
# ---------------------------------------------------------------------------

_MIT_TEMPLATE = textwrap.dedent("""\
    MIT License

    Copyright (c) {year} {author}

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
""")

_APACHE2_TEMPLATE = textwrap.dedent("""\
    Apache License, Version 2.0

    Copyright {year} {author}

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
""")

_LICENSE_TEMPLATES: dict[str, str] = {
    "MIT": _MIT_TEMPLATE,
    "Apache-2.0": _APACHE2_TEMPLATE,
}

_GITIGNORE = textwrap.dedent("""\
    __pycache__/
    *.py[cod]
    *$py.class
    *.so
    dist/
    build/
    *.egg-info/
    .eggs/
    *.egg
    .venv/
    venv/
    .env
    .mypy_cache/
    .ruff_cache/
""")


def _sanitize_class_name(name: str) -> str:
    """Convert a kebab/snake string to PascalCase class name.

    Args:
        name: Raw name string.

    Returns:
        PascalCase version suitable as a Python class identifier.
    """
    parts = re.split(r"[-_ ]+", name)
    return "".join(p.capitalize() for p in parts if p)


def _extract_node_classes(source: str) -> list[str]:
    """Extract class names that look like ComfyUI node definitions.

    Scans for classes that define a ``CATEGORY`` or ``FUNCTION`` attribute,
    which is the hallmark of a ComfyUI node class.

    Args:
        source: Python source code.

    Returns:
        List of class names identified as node definitions.
    """
    classes: list[str] = []
    current_class: str | None = None
    for line in source.splitlines():
        stripped = line.strip()
        match = re.match(r"^class\s+(\w+)", stripped)
        if match:
            current_class = match.group(1)
        elif current_class and re.match(r"(CATEGORY|FUNCTION)\s*=", stripped):
            if current_class not in classes:
                classes.append(current_class)
            current_class = None
    return classes


class PackageScaffolder:
    """Generate a publishable ComfyUI node package directory.

    Given generated node source code and a ``PackageConfig``, produces the
    full directory tree expected by the Comfy Registry and ComfyUI Manager.
    """

    def scaffold(
        self,
        node_code_path: str,
        config: PackageConfig,
        output_dir: str,
    ) -> str:
        """Create the full package directory structure.

        Args:
            node_code_path: Path to the generated node ``.py`` file.
            config: Package metadata configuration.
            output_dir: Root directory where the package folder is created.

        Returns:
            Absolute path to the created package directory.

        Raises:
            FileNotFoundError: If *node_code_path* does not exist.
            ValueError: If no ComfyUI node classes can be found in source.
        """
        source_path = Path(node_code_path)
        if not source_path.exists():
            msg = f"Node source file not found: {node_code_path}"
            raise FileNotFoundError(msg)

        source = source_path.read_text(encoding="utf-8")
        node_classes = _extract_node_classes(source)
        if not node_classes:
            msg = "No ComfyUI node classes found in source file"
            raise ValueError(msg)

        pkg_dir = Path(output_dir) / config.name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Copy node source
        (pkg_dir / source_path.name).write_text(source, encoding="utf-8")

        node_info = self._build_node_info(node_classes, source)

        # Generate scaffolded files
        (pkg_dir / "__init__.py").write_text(
            self._generate_init(node_classes, source_path.stem),
            encoding="utf-8",
        )
        (pkg_dir / "pyproject.toml").write_text(
            self._generate_pyproject(config),
            encoding="utf-8",
        )
        (pkg_dir / "requirements.txt").write_text(
            self._generate_requirements(config.dependencies),
            encoding="utf-8",
        )
        (pkg_dir / "README.md").write_text(
            self._generate_readme(config, node_info),
            encoding="utf-8",
        )
        (pkg_dir / "LICENSE").write_text(
            self._generate_license(config),
            encoding="utf-8",
        )
        (pkg_dir / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")

        return str(pkg_dir.resolve())

    # ------------------------------------------------------------------
    # Internal generators
    # ------------------------------------------------------------------

    @staticmethod
    def _build_node_info(
        node_classes: list[str],
        source: str,
    ) -> dict[str, Any]:
        """Build a summary dict used by README generation.

        Args:
            node_classes: Discovered node class names.
            source: Full node source code.

        Returns:
            Dictionary with ``classes`` and ``source_lines`` keys.
        """
        return {
            "classes": node_classes,
            "source_lines": len(source.splitlines()),
        }

    @staticmethod
    def _generate_init(node_classes: list[str], module_name: str) -> str:
        """Generate ``__init__.py`` with node mapping exports.

        Args:
            node_classes: List of node class names to register.
            module_name: Python module name (stem of the source file).

        Returns:
            String content for ``__init__.py``.
        """
        imports = ", ".join(node_classes)
        class_mappings = ", ".join(f'"{cls}": {cls}' for cls in node_classes)
        display_mappings = ", ".join(
            f'"{cls}": "{_sanitize_class_name(cls)}"' for cls in node_classes
        )

        return textwrap.dedent(f"""\
            \"\"\"ComfyUI custom node package.\"\"\"
            from .{module_name} import {imports}

            NODE_CLASS_MAPPINGS = {{{class_mappings}}}
            NODE_DISPLAY_NAME_MAPPINGS = {{{display_mappings}}}

            __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
        """)

    @staticmethod
    def _generate_pyproject(config: PackageConfig) -> str:
        """Generate ``pyproject.toml`` with ``[tool.comfy]`` section.

        Args:
            config: Package configuration.

        Returns:
            String content for ``pyproject.toml``.
        """
        deps_list = ", ".join(f'"{d}"' for d in config.dependencies)
        tags_list = ", ".join(f'"{t}"' for t in config.tags)

        return textwrap.dedent(f"""\
            [project]
            name = "{config.name}"
            version = "{config.version}"
            description = "{config.description}"
            license = "{config.license}"
            requires-python = ">=3.10"
            dependencies = [{deps_list}]

            [project.urls]
            Repository = "{config.repository_url}"

            [tool.comfy]
            PublisherId = ""
            DisplayName = "{_sanitize_class_name(config.name)}"
            Icon = ""
            Tags = [{tags_list}]
        """)

    @staticmethod
    def _generate_readme(config: PackageConfig, node_info: dict[str, Any]) -> str:
        """Generate ``README.md`` with node documentation.

        Args:
            config: Package configuration.
            node_info: Node summary from ``_build_node_info``.

        Returns:
            String content for ``README.md``.
        """
        classes_list = "\n".join(f"- `{cls}`" for cls in node_info.get("classes", []))

        install_section = textwrap.dedent(f"""\
            ## Installation

            ### Via ComfyUI Manager

            Search for **{config.name}** in ComfyUI Manager and click Install.

            ### Via Comfy Registry

            ```bash
            comfy node registry-install {config.name}
            ```

            ### Manual

            ```bash
            cd ComfyUI/custom_nodes
            git clone {config.repository_url or "https://github.com/your/repo.git"}
            pip install -r requirements.txt
            ```
        """)

        return textwrap.dedent(f"""\
            # {config.name}

            {config.description}

            ## Nodes

            {classes_list}

            {install_section}
            ## License

            {config.license}
        """)

    @staticmethod
    def _generate_requirements(dependencies: list[str]) -> str:
        """Generate ``requirements.txt`` with pinned deps.

        Args:
            dependencies: List of pip requirement strings.

        Returns:
            String content for ``requirements.txt``.
        """
        if not dependencies:
            return "# No additional dependencies\n"
        return "\n".join(dependencies) + "\n"

    @staticmethod
    def _generate_license(config: PackageConfig) -> str:
        """Generate license file content.

        Args:
            config: Package configuration (uses ``license`` and ``author``).

        Returns:
            License text with author and year substituted.
        """
        template = _LICENSE_TEMPLATES.get(config.license, _MIT_TEMPLATE)
        year = datetime.now(tz=timezone.utc).strftime("%Y")
        return template.format(year=year, author=config.author or "Contributors")
