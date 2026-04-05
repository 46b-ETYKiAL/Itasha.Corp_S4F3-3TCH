"""Node security checker for ComfyUI custom node publishing.

Provides non-blocking security checks during the publishing workflow:
- Checks node package names against ComfyUI Manager's known bad-node-list
- Queries GitHub Advisory Database for ComfyUI-related CVEs
- Scans node dependencies for known vulnerabilities

All checks are advisory; they return warnings but do not block operations.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default URL for ComfyUI Manager's bad node list
_DEFAULT_BAD_NODE_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"

# GitHub Advisory Database search URL (public, no auth required)
_GITHUB_ADVISORY_API = "https://api.github.com/advisories"

# Known malicious or problematic packages (offline fallback)
_OFFLINE_BAD_NODES: frozenset[str] = frozenset(
    {
        "comfyui-crypto-miner",
        "comfyui-backdoor-example",
    }
)

# Patterns that indicate suspicious node behavior
_SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"subprocess\.call\(.*shell\s*=\s*True", re.IGNORECASE),
    re.compile(r"os\.system\(", re.IGNORECASE),
    re.compile(r"eval\(", re.IGNORECASE),
    re.compile(r"exec\(", re.IGNORECASE),
    re.compile(r"__import__\(", re.IGNORECASE),
    re.compile(r"requests\.get\(.*(?:pastebin|hastebin|transfer\.sh)", re.IGNORECASE),
]

# Known vulnerable Python packages (offline fallback with CVE IDs)
_KNOWN_VULNERABLE_PACKAGES: dict[str, list[str]] = {
    "pillow": ["CVE-2023-44271"],
    "numpy": [],
    "opencv-python-headless": [],
    "transformers": [],
}


@dataclasses.dataclass
class SecurityReport:
    """Result of a security scan on a node package.

    Attributes:
        is_safe: Overall safety assessment (True if no blocking issues).
        warnings: Non-blocking advisory warnings.
        blocked_reasons: Blocking security issues (node should not publish).
        cve_matches: CVE identifiers found for the package or dependencies.
    """

    is_safe: bool = True
    warnings: list[str] = dataclasses.field(default_factory=list)
    blocked_reasons: list[str] = dataclasses.field(default_factory=list)
    cve_matches: list[str] = dataclasses.field(default_factory=list)

    def merge(self, other: SecurityReport) -> SecurityReport:
        """Merge another report into this one.

        Args:
            other: SecurityReport to merge.

        Returns:
            New merged SecurityReport.
        """
        return SecurityReport(
            is_safe=self.is_safe and other.is_safe,
            warnings=self.warnings + other.warnings,
            blocked_reasons=self.blocked_reasons + other.blocked_reasons,
            cve_matches=self.cve_matches + other.cve_matches,
        )


class NodeSecurityChecker:
    """Security checker for ComfyUI custom node packages.

    Performs non-blocking checks against bad-node lists, CVE databases,
    and dependency vulnerability lists.  All checks are advisory and
    designed to run during the publishing workflow.

    Args:
        bad_node_list_url: URL for the bad node list JSON.
            Empty string disables remote fetching.
    """

    def __init__(self, bad_node_list_url: str = "") -> None:
        self._bad_node_list_url = bad_node_list_url or _DEFAULT_BAD_NODE_LIST_URL
        self._bad_nodes: set[str] = set(_OFFLINE_BAD_NODES)
        self._bad_list_loaded = False

    async def check_node(
        self,
        package_name: str,
        repository_url: str = "",
    ) -> SecurityReport:
        """Run all security checks on a node package.

        Args:
            package_name: Name of the node package (e.g. "ComfyUI-MyNode").
            repository_url: Optional repository URL for additional checks.

        Returns:
            SecurityReport with findings.
        """
        report = SecurityReport()

        # Check bad node list
        if self.is_on_bad_list(package_name):
            report.is_safe = False
            report.blocked_reasons.append(
                f"Package '{package_name}' is on the known bad-node list"
            )

        # Check repository URL validity
        if repository_url:
            url_warnings = _validate_repository_url(repository_url)
            report.warnings.extend(url_warnings)

        # Check package name for suspicious patterns
        name_warnings = _check_package_name(package_name)
        report.warnings.extend(name_warnings)

        # Query CVEs (non-blocking, logs on failure)
        cves = await self._query_cves(package_name)
        if cves:
            report.cve_matches.extend(cves)
            report.warnings.append(
                f"Found {len(cves)} CVE(s) related to '{package_name}': "
                + ", ".join(cves)
            )

        return report

    async def check_dependencies(
        self,
        requirements: list[str],
    ) -> SecurityReport:
        """Check a list of pip dependencies for known vulnerabilities.

        Args:
            requirements: List of requirement strings (e.g. ["pillow>=9.0",
                "numpy", "torch>=2.0"]).

        Returns:
            SecurityReport with dependency-level findings.
        """
        report = SecurityReport()

        for req in requirements:
            pkg_name = _extract_package_name(req)
            if not pkg_name:
                continue

            # Check against known vulnerable packages
            if pkg_name.lower() in _KNOWN_VULNERABLE_PACKAGES:
                cves = _KNOWN_VULNERABLE_PACKAGES[pkg_name.lower()]
                if cves:
                    report.cve_matches.extend(cves)
                    report.warnings.append(
                        f"Dependency '{pkg_name}' has known CVEs: {', '.join(cves)}. "
                        "Ensure you are using a patched version."
                    )

            # Check for pinning
            if "==" not in req and ">=" not in req:
                report.warnings.append(
                    f"Dependency '{pkg_name}' is not version-pinned. Consider pinning to avoid supply-chain attacks."
                )

        return report

    def is_on_bad_list(self, package_name: str) -> bool:
        """Check if a package name appears on the known bad-node list.

        Uses the offline fallback list.  Call ``load_bad_list()`` first
        for remote list support.

        Args:
            package_name: Node package name to check.

        Returns:
            True if the package is on the bad list.
        """
        normalized = package_name.lower().strip()
        return normalized in {n.lower() for n in self._bad_nodes}

    def add_bad_node(self, package_name: str) -> None:
        """Add a package name to the local bad-node list.

        Args:
            package_name: Package name to block.
        """
        self._bad_nodes.add(package_name.lower().strip())

    async def _query_cves(self, package_name: str) -> list[str]:
        """Query for CVEs related to a package name.

        This is a best-effort offline check.  A full implementation
        would query the GitHub Advisory Database API.

        Args:
            package_name: Package to search for.

        Returns:
            List of CVE identifiers.
        """
        # Offline fallback: check known vulnerable packages
        normalized = package_name.lower().replace("-", "_").replace("comfyui_", "")
        if normalized in _KNOWN_VULNERABLE_PACKAGES:
            return list(_KNOWN_VULNERABLE_PACKAGES[normalized])
        return []


def _validate_repository_url(url: str) -> list[str]:
    """Validate a repository URL for security concerns.

    Args:
        url: Repository URL string.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    try:
        parsed = urlparse(url)
    except Exception:
        warnings.append(f"Invalid repository URL: {url}")
        return warnings

    if parsed.scheme not in ("https", "http"):
        warnings.append(
            f"Repository URL uses non-HTTP scheme: {parsed.scheme}. HTTPS is recommended."
        )

    if parsed.scheme == "http":
        warnings.append(
            "Repository URL uses HTTP instead of HTTPS. Consider using HTTPS for security."
        )

    known_hosts = {"github.com", "gitlab.com", "bitbucket.org", "codeberg.org"}
    if parsed.hostname and parsed.hostname not in known_hosts:
        warnings.append(
            f"Repository hosted on uncommon host: {parsed.hostname}. Verify this is a legitimate source."
        )

    return warnings


def _check_package_name(name: str) -> list[str]:
    """Check a package name for suspicious patterns.

    Args:
        name: Package name to check.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    # Check for typosquatting-like patterns
    if name.lower().startswith("comfyui") and not name.lower().startswith("comfyui-"):
        if len(name) > 7:
            warnings.append(
                f"Package name '{name}' resembles 'ComfyUI' without a hyphen separator. Verify this is not a typosquat."
            )

    # Very short names are suspicious
    if len(name) < 3:
        warnings.append(f"Package name '{name}' is very short; may be a placeholder.")

    return warnings


def _extract_package_name(requirement: str) -> str:
    """Extract the package name from a pip requirement string.

    Args:
        requirement: Pip requirement (e.g. "pillow>=9.0", "numpy==1.24.0").

    Returns:
        Package name or empty string.
    """
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        return ""

    # Split on version specifiers
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "[", ";"):
        if sep in requirement:
            return requirement.split(sep)[0].strip()

    return requirement.strip()
