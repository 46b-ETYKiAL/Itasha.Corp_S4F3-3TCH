"""SemVer version management for ComfyUI node packages.

Handles version parsing, bumping, dependency conflict detection
against well-known ComfyUI ecosystem packages, and changelog
generation.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Well-known ComfyUI ecosystem packages (top 50 by install count).
# Used for dependency conflict warnings — not an exhaustive list.
# ---------------------------------------------------------------------------

KNOWN_PACKAGES: dict[str, str] = {
    "torch": ">=2.1.0",
    "torchvision": ">=0.16.0",
    "torchaudio": ">=2.1.0",
    "numpy": ">=1.24.0",
    "Pillow": ">=9.5.0",
    "scipy": ">=1.10.0",
    "opencv-python": ">=4.7.0",
    "opencv-python-headless": ">=4.7.0",
    "transformers": ">=4.30.0",
    "safetensors": ">=0.3.1",
    "accelerate": ">=0.20.0",
    "diffusers": ">=0.25.0",
    "einops": ">=0.6.0",
    "kornia": ">=0.7.0",
    "scikit-image": ">=0.20.0",
    "requests": ">=2.28.0",
    "tqdm": ">=4.64.0",
    "aiohttp": ">=3.8.0",
    "pyyaml": ">=6.0",
    "psutil": ">=5.9.0",
    "huggingface-hub": ">=0.20.0",
    "tokenizers": ">=0.13.0",
    "onnxruntime": ">=1.14.0",
    "onnxruntime-gpu": ">=1.14.0",
    "xformers": ">=0.0.20",
    "triton": ">=2.0.0",
    "timm": ">=0.9.0",
    "segment-anything": ">=1.0",
    "ultralytics": ">=8.0.0",
    "insightface": ">=0.7.0",
    "mediapipe": ">=0.10.0",
    "color-matcher": ">=0.5.0",
    "rembg": ">=2.0.50",
    "facexlib": ">=0.3.0",
    "gfpgan": ">=1.3.0",
    "realesrgan": ">=0.3.0",
    "basicsr": ">=1.4.0",
    "lark": ">=1.1.0",
    "jsonschema": ">=4.17.0",
    "gitpython": ">=3.1.30",
    "toml": ">=0.10.0",
    "tomli": ">=2.0.0",
    "typing-extensions": ">=4.5.0",
    "filelock": ">=3.9.0",
    "regex": ">=2022.10.0",
    "ftfy": ">=6.1.0",
    "open-clip-torch": ">=2.20.0",
    "clip-interrogator": ">=0.6.0",
    "controlnet-aux": ">=0.0.7",
    "depth-anything": ">=1.0.0",
}

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-.]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z\-.]+))?$"
)

_VALID_BUMP_TYPES = frozenset({"patch", "minor", "major"})


class VersionError(ValueError):
    """Raised when a version string is invalid."""


class VersionManager:
    """SemVer version parsing, bumping, and changelog utilities."""

    def parse_version(self, version_str: str) -> tuple[int, int, int]:
        """Parse a SemVer string into (major, minor, patch).

        Args:
            version_str: A version string like ``"1.2.3"``.

        Returns:
            Tuple of (major, minor, patch) integers.

        Raises:
            VersionError: If the string is not valid SemVer.
        """
        match = _SEMVER_RE.match(version_str)
        if not match:
            msg = f"Invalid SemVer string: {version_str!r}"
            raise VersionError(msg)
        return (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
        )

    def bump(self, current: str, bump_type: str = "patch") -> str:
        """Bump a version string by the specified component.

        Args:
            current: Current SemVer version.
            bump_type: One of ``"patch"``, ``"minor"``, ``"major"``.

        Returns:
            New version string after the bump.

        Raises:
            VersionError: If *current* is not valid SemVer.
            ValueError: If *bump_type* is not recognized.
        """
        if bump_type not in _VALID_BUMP_TYPES:
            msg = f"bump_type must be one of {sorted(_VALID_BUMP_TYPES)}, got: {bump_type!r}"
            raise ValueError(msg)

        major, minor, patch = self.parse_version(current)

        if bump_type == "major":
            return f"{major + 1}.0.0"
        if bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    def check_conflicts(self, dependencies: list[str]) -> list[str]:
        """Check dependencies for potential conflicts with known packages.

        Detects cases where a dependency names a well-known ComfyUI
        ecosystem package, which could cause version conflicts if the
        user's constraint is incompatible.

        Args:
            dependencies: List of pip requirement strings.

        Returns:
            List of human-readable conflict warnings. Empty if none.
        """
        warnings: list[str] = []
        for dep in dependencies:
            name = _extract_package_name(dep)
            if name in KNOWN_PACKAGES:
                known_constraint = KNOWN_PACKAGES[name]
                warnings.append(
                    f"{name} is a core ComfyUI ecosystem package "
                    f"(known constraint: {known_constraint}). "
                    f"Your requirement '{dep}' may conflict with existing installations."
                )
        return warnings

    def generate_changelog_entry(
        self,
        version: str,
        changes: list[str],
        *,
        date: datetime | None = None,
    ) -> str:
        """Generate a markdown changelog entry.

        Args:
            version: Version string for this entry.
            changes: List of change descriptions.
            date: Entry date. Defaults to current UTC date.

        Returns:
            Markdown-formatted changelog entry string.
        """
        if date is None:
            date = datetime.now(tz=timezone.utc)
        date_str = date.strftime("%Y-%m-%d")

        lines = [f"## [{version}] - {date_str}", ""]
        for change in changes:
            lines.append(f"- {change}")
        lines.append("")
        return "\n".join(lines)


def _extract_package_name(requirement: str) -> str:
    """Extract the bare package name from a pip requirement string.

    Args:
        requirement: A pip requirement like ``"numpy>=1.24.0"``.

    Returns:
        Bare package name (e.g. ``"numpy"``).
    """
    match = re.match(r"^([a-zA-Z0-9_\-]+)", requirement)
    return match.group(1) if match else requirement
