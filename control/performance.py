"""Performance auto-tuner for ComfyUI.

Detects GPU hardware, recommends attention method and VRAM mode,
checks for optional acceleration libraries, and generates ComfyUI
command-line flags for optimal performance.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# GPU architecture detection patterns (NVIDIA)
_ARCH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"RTX\s*50[789]0|RTX\s*5060|Blackwell", re.IGNORECASE), "blackwell"),
    (
        re.compile(r"RTX\s*40[789]0|RTX\s*4060|Ada|L40|L4", re.IGNORECASE),
        "ada_lovelace",
    ),
    (re.compile(r"H100|H200|GH200|Hopper", re.IGNORECASE), "hopper"),
    (
        re.compile(r"RTX\s*30[789]0|RTX\s*3060|RTX\s*A[456]000|Ampere", re.IGNORECASE),
        "ampere",
    ),
    (re.compile(r"RTX\s*20[789]0|RTX\s*2060|Turing|T4", re.IGNORECASE), "turing"),
    (re.compile(r"GTX\s*10[789]0|GTX\s*1060|Pascal|P100|P40", re.IGNORECASE), "pascal"),
    (re.compile(r"V100|Volta", re.IGNORECASE), "volta"),
]

# Architectures that support SageAttention efficiently
_SAGE_ARCHITECTURES = frozenset({"ampere", "ada_lovelace", "hopper", "blackwell"})

# VRAM thresholds for mode recommendations (MB)
_HIGH_VRAM_THRESHOLD = 12288  # 12GB
_LOW_VRAM_THRESHOLD = 6144  # 6GB
_NOVRAM_THRESHOLD = 3072  # 3GB


@dataclasses.dataclass
class GPUInfo:
    """Detected GPU hardware information.

    Attributes:
        name: GPU model name (e.g. "NVIDIA GeForce RTX 4090").
        vram_mb: Total VRAM in megabytes.
        cuda_version: CUDA toolkit version string.
        pytorch_version: PyTorch version if available.
        driver_version: NVIDIA driver version string.
        architecture: GPU architecture family (ampere, ada_lovelace, etc.).
    """

    name: str = "unknown"
    vram_mb: int = 0
    cuda_version: str = ""
    pytorch_version: str = ""
    driver_version: str = ""
    architecture: str = ""


@dataclasses.dataclass
class PerformanceRecommendation:
    """Recommended ComfyUI performance configuration.

    Attributes:
        attention_method: Attention implementation to use
            (sdp, sage, flash, xformers).
        vram_mode: VRAM management mode
            (highvram, normal, lowvram, novram).
        preview_method: Preview generation method (auto, taesd, latent).
        extra_flags: Additional ComfyUI CLI flags.
        notes: Human-readable explanation of recommendations.
    """

    attention_method: str = "sdp"
    vram_mode: str = "normal"
    preview_method: str = "auto"
    extra_flags: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)


class PerformanceTuner:
    """Auto-tuner that detects hardware and recommends ComfyUI settings.

    Probes the system for GPU information using ``nvidia-smi`` and optional
    PyTorch imports, then generates performance recommendations including
    attention method, VRAM mode, and CLI flags.
    """

    def detect_gpu(self) -> GPUInfo:
        """Detect GPU hardware information.

        Tries ``nvidia-smi`` first, then falls back to PyTorch CUDA
        introspection.  Returns a GPUInfo with best-effort populated
        fields.

        Returns:
            GPUInfo with detected hardware details.
        """
        info = GPUInfo()

        # Try nvidia-smi first
        info = self._detect_via_nvidia_smi(info)

        # Enrich with PyTorch if available
        info = self._detect_via_pytorch(info)

        # Determine architecture from GPU name
        if info.name != "unknown" and not info.architecture:
            info = dataclasses.replace(
                info, architecture=_detect_architecture(info.name)
            )

        return info

    def recommend(self, gpu: GPUInfo | None = None) -> PerformanceRecommendation:
        """Generate performance recommendations for the detected GPU.

        Args:
            gpu: Pre-detected GPU info.  If None, runs detection first.

        Returns:
            PerformanceRecommendation with optimal settings.
        """
        if gpu is None:
            gpu = self.detect_gpu()

        rec = PerformanceRecommendation()

        # Attention method
        rec = self._recommend_attention(gpu, rec)

        # VRAM mode
        rec = self._recommend_vram_mode(gpu, rec)

        # Preview method
        if gpu.vram_mb > 0 and gpu.vram_mb < _LOW_VRAM_THRESHOLD:
            rec = dataclasses.replace(
                rec,
                preview_method="taesd",
                notes=[*rec.notes, "Using TAESD previews to save VRAM"],
            )

        return rec

    def check_sage_attention(self) -> bool:
        """Check if SageAttention is installed and importable.

        Returns:
            True if SageAttention is available.
        """
        return importlib.util.find_spec("sageattention") is not None

    def check_flash_attention(self) -> bool:
        """Check if Flash Attention is installed and importable.

        Returns:
            True if flash-attn is available.
        """
        return importlib.util.find_spec("flash_attn") is not None

    def check_xformers(self) -> bool:
        """Check if xformers is installed and importable.

        Returns:
            True if xformers is available.
        """
        return importlib.util.find_spec("xformers") is not None

    def get_comfyui_flags(
        self,
        recommendation: PerformanceRecommendation,
    ) -> list[str]:
        """Convert a recommendation into ComfyUI CLI flags.

        Args:
            recommendation: Performance recommendation to convert.

        Returns:
            List of CLI flag strings (e.g. ["--use-sage-attention",
            "--highvram"]).
        """
        flags: list[str] = []

        # Attention flags
        attention_flags: dict[str, str] = {
            "sage": "--use-sage-attention",
            "flash": "--use-flash-attention",
            "xformers": "--use-xformers",
        }
        if recommendation.attention_method in attention_flags:
            flags.append(attention_flags[recommendation.attention_method])

        # VRAM mode flags
        vram_flags: dict[str, str] = {
            "highvram": "--highvram",
            "lowvram": "--lowvram",
            "novram": "--novram",
        }
        if recommendation.vram_mode in vram_flags:
            flags.append(vram_flags[recommendation.vram_mode])

        # Preview method
        if recommendation.preview_method == "taesd":
            flags.append("--preview-method")
            flags.append("taesd")

        # Extra flags
        flags.extend(recommendation.extra_flags)

        return flags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_via_nvidia_smi(self, info: GPUInfo) -> GPUInfo:
        """Detect GPU info using nvidia-smi.

        Args:
            info: Existing GPUInfo to enrich.

        Returns:
            Updated GPUInfo.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    name = parts[0]
                    vram_mb = int(float(parts[1]))
                    driver = parts[2]
                    info = dataclasses.replace(
                        info,
                        name=name,
                        vram_mb=vram_mb,
                        driver_version=driver,
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.debug("nvidia-smi not available; falling back to PyTorch detection")

        # Try to get CUDA version
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0:
                # nvidia-smi doesn't directly report CUDA version in query mode;
                # parse from the main output instead
                full_result = subprocess.run(
                    ["nvidia-smi"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False,
                )
                if full_result.returncode == 0:
                    cuda_match = re.search(
                        r"CUDA Version:\s*([\d.]+)", full_result.stdout
                    )
                    if cuda_match:
                        info = dataclasses.replace(
                            info, cuda_version=cuda_match.group(1)
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return info

    def _detect_via_pytorch(self, info: GPUInfo) -> GPUInfo:
        """Detect GPU info using PyTorch CUDA APIs.

        Args:
            info: Existing GPUInfo to enrich.

        Returns:
            Updated GPUInfo.
        """
        try:
            import torch

            info = dataclasses.replace(info, pytorch_version=torch.__version__)

            if torch.cuda.is_available():
                if info.name == "unknown":
                    info = dataclasses.replace(info, name=torch.cuda.get_device_name(0))
                if info.vram_mb == 0:
                    total = torch.cuda.get_device_properties(0).total_mem
                    info = dataclasses.replace(info, vram_mb=int(total / (1024 * 1024)))
                if not info.cuda_version:
                    info = dataclasses.replace(
                        info, cuda_version=torch.version.cuda or ""
                    )
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
        except Exception:
            logger.debug("PyTorch CUDA detection failed", exc_info=True)

        return info

    def _recommend_attention(
        self,
        gpu: GPUInfo,
        rec: PerformanceRecommendation,
    ) -> PerformanceRecommendation:
        """Recommend attention method based on GPU.

        Args:
            gpu: Detected GPU info.
            rec: Existing recommendation to update.

        Returns:
            Updated PerformanceRecommendation.
        """
        # SageAttention preferred for Ampere+ if installed
        if gpu.architecture in _SAGE_ARCHITECTURES and self.check_sage_attention():
            return dataclasses.replace(
                rec,
                attention_method="sage",
                notes=[
                    *rec.notes,
                    f"SageAttention recommended for {gpu.architecture} architecture",
                ],
            )

        # Flash Attention for training workloads or when sage is unavailable
        if self.check_flash_attention():
            return dataclasses.replace(
                rec,
                attention_method="flash",
                notes=[*rec.notes, "Flash Attention available as fallback"],
            )

        # xformers as secondary fallback
        if self.check_xformers():
            return dataclasses.replace(
                rec,
                attention_method="xformers",
                notes=[*rec.notes, "Using xformers for memory-efficient attention"],
            )

        # Default SDP (built into PyTorch 2.0+)
        return dataclasses.replace(
            rec,
            attention_method="sdp",
            notes=[*rec.notes, "Using PyTorch SDP attention (default)"],
        )

    def _recommend_vram_mode(
        self,
        gpu: GPUInfo,
        rec: PerformanceRecommendation,
    ) -> PerformanceRecommendation:
        """Recommend VRAM management mode.

        Args:
            gpu: Detected GPU info.
            rec: Existing recommendation to update.

        Returns:
            Updated PerformanceRecommendation.
        """
        if gpu.vram_mb <= 0:
            return dataclasses.replace(
                rec,
                vram_mode="normal",
                notes=[
                    *rec.notes,
                    "Could not detect VRAM; using normal mode",
                ],
            )

        if gpu.vram_mb >= _HIGH_VRAM_THRESHOLD:
            return dataclasses.replace(
                rec,
                vram_mode="highvram",
                notes=[
                    *rec.notes,
                    f"High VRAM mode: {gpu.vram_mb}MB available (>={_HIGH_VRAM_THRESHOLD}MB)",
                ],
            )

        if gpu.vram_mb < _NOVRAM_THRESHOLD:
            return dataclasses.replace(
                rec,
                vram_mode="novram",
                notes=[
                    *rec.notes,
                    f"No-VRAM mode: only {gpu.vram_mb}MB available (<{_NOVRAM_THRESHOLD}MB)",
                ],
            )

        if gpu.vram_mb < _LOW_VRAM_THRESHOLD:
            return dataclasses.replace(
                rec,
                vram_mode="lowvram",
                notes=[
                    *rec.notes,
                    f"Low VRAM mode: {gpu.vram_mb}MB available (<{_LOW_VRAM_THRESHOLD}MB)",
                ],
            )

        return dataclasses.replace(
            rec,
            vram_mode="normal",
            notes=[
                *rec.notes,
                f"Normal VRAM mode: {gpu.vram_mb}MB available",
            ],
        )


def _detect_architecture(gpu_name: str) -> str:
    """Detect GPU architecture from its name string.

    Args:
        gpu_name: GPU model name from nvidia-smi or PyTorch.

    Returns:
        Architecture family string or "unknown".
    """
    for pattern, arch in _ARCH_PATTERNS:
        if pattern.search(gpu_name):
            return arch
    return "unknown"
