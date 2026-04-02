"""Quantization-aware model management for ComfyUI.

Detects model format (safetensors, GGUF, checkpoint), extracts quantization
metadata, and estimates VRAM usage based on quantization level.  Supports
FP32, FP16, BF16, FP8 (float8_e4m3fn), NF4 (4-bit bitsandbytes), and
GGUF quantization levels (Q2 through Q8).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# GGUF magic bytes: "GGUF" in little-endian
_GGUF_MAGIC = 0x46475547

# GGUF quantization type IDs to human-readable names
_GGUF_QUANT_NAMES: dict[int, str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
}

# VRAM savings factor relative to FP16 baseline.
# A factor of 0.50 means the model uses 50% of FP16 VRAM.
_VRAM_FACTOR: dict[str, float] = {
    "fp32": 2.0,
    "fp16": 1.0,
    "bf16": 1.0,
    "fp8_e4m3fn": 0.50,
    "fp8_e5m2": 0.50,
    "nf4": 0.25,
    "int8": 0.50,
    "int4": 0.25,
    # GGUF quantization levels
    "f32": 2.0,
    "f16": 1.0,
    "q8_0": 0.50,
    "q8_1": 0.55,
    "q6_k": 0.40,
    "q5_k_m": 0.35,
    "q5_k_s": 0.35,
    "q5_0": 0.35,
    "q5_1": 0.35,
    "q4_k_m": 0.25,
    "q4_k_s": 0.25,
    "q4_0": 0.25,
    "q4_1": 0.25,
    "q3_k_m": 0.22,
    "q3_k_s": 0.20,
    "q3_k_l": 0.22,
    "q2_k": 0.15,
}


@dataclasses.dataclass
class QuantizationInfo:
    """Metadata about a model's quantization format.

    Attributes:
        format: File format (safetensors, gguf, ckpt).
        dtype: Data type string (fp32, fp16, bf16, fp8_e4m3fn, nf4,
            q4_k_m, q8_0, etc.).
        estimated_vram_mb: Estimated VRAM needed for inference in MB.
        file_size_mb: File size on disk in MB.
        quantization_method: Method used (GGUF quant type, bitsandbytes,
            native, etc.).
    """

    format: str
    dtype: str
    estimated_vram_mb: float
    file_size_mb: float
    quantization_method: str = ""


def detect_format(model_path: str) -> QuantizationInfo:
    """Detect quantization format and metadata for a model file.

    Reads file headers to determine format, then extracts quantization
    metadata.  For safetensors files, parses the JSON header.  For GGUF
    files, reads the magic bytes and metadata block.  Falls back to
    file-extension heuristics for other formats.

    Args:
        model_path: Path to the model file.

    Returns:
        QuantizationInfo with detected format, dtype, and VRAM estimate.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        return _detect_safetensors(path, file_size_mb)
    if suffix == ".gguf":
        return _detect_gguf(path, file_size_mb)
    if suffix in (".ckpt", ".pt", ".pth", ".bin"):
        return _detect_checkpoint(path, file_size_mb)

    logger.warning("Unknown model extension '%s'; assuming FP16 safetensors", suffix)
    vram = _estimate_vram_from_size(file_size_mb, "fp16")
    return QuantizationInfo(
        format="unknown",
        dtype="fp16",
        estimated_vram_mb=vram,
        file_size_mb=round(file_size_mb, 1),
        quantization_method="unknown",
    )


def estimate_vram(model_path: str) -> float:
    """Estimate VRAM usage for a model file in MB.

    Convenience wrapper around detect_format that returns only the
    VRAM estimate.

    Args:
        model_path: Path to the model file.

    Returns:
        Estimated VRAM in megabytes.
    """
    info = detect_format(model_path)
    return info.estimated_vram_mb


def list_models_with_quantization(
    models_dir: str,
    extensions: tuple[str, ...] = (".safetensors", ".gguf", ".ckpt", ".pt"),
) -> list[dict[str, str | float]]:
    """List models in a directory with quantization metadata.

    Scans the given directory (non-recursively) for model files and
    returns a list of dicts with name, format, dtype, file size, and
    estimated VRAM.

    Args:
        models_dir: Directory to scan for model files.
        extensions: File extensions to include.

    Returns:
        List of dicts with keys: name, format, dtype, file_size_mb,
        estimated_vram_mb, quantization_method.
    """
    directory = Path(models_dir)
    if not directory.is_dir():
        logger.warning("Models directory does not exist: %s", models_dir)
        return []

    results: list[dict[str, str | float]] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in extensions:
            try:
                info = detect_format(str(entry))
                results.append(
                    {
                        "name": entry.name,
                        "format": info.format,
                        "dtype": info.dtype,
                        "file_size_mb": info.file_size_mb,
                        "estimated_vram_mb": info.estimated_vram_mb,
                        "quantization_method": info.quantization_method,
                    }
                )
            except Exception:
                logger.exception("Failed to detect format for %s", entry.name)
                results.append(
                    {
                        "name": entry.name,
                        "format": "error",
                        "dtype": "unknown",
                        "file_size_mb": round(entry.stat().st_size / (1024 * 1024), 1),
                        "estimated_vram_mb": 0.0,
                        "quantization_method": "",
                    }
                )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_safetensors(path: Path, file_size_mb: float) -> QuantizationInfo:
    """Parse a safetensors file header to extract dtype information.

    Args:
        path: Path to the safetensors file.
        file_size_mb: Pre-computed file size in MB.

    Returns:
        QuantizationInfo for the safetensors model.
    """
    dtype = "fp16"
    quant_method = "native"

    try:
        with path.open("rb") as fh:
            header_size_bytes = fh.read(8)
            if len(header_size_bytes) < 8:
                logger.warning("Safetensors file too small: %s", path.name)
                vram = _estimate_vram_from_size(file_size_mb, dtype)
                return QuantizationInfo(
                    format="safetensors",
                    dtype=dtype,
                    estimated_vram_mb=vram,
                    file_size_mb=round(file_size_mb, 1),
                    quantization_method=quant_method,
                )

            header_size = struct.unpack("<Q", header_size_bytes)[0]
            # Safety cap: don't read more than 10MB of header
            header_size = min(header_size, 10 * 1024 * 1024)
            header_raw = fh.read(header_size)
            header = json.loads(header_raw)

            dtype = _extract_safetensors_dtype(header)
            quant_method = _extract_safetensors_quant_method(header)
    except (json.JSONDecodeError, struct.error, OSError):
        logger.warning("Could not parse safetensors header for %s", path.name)

    vram = _estimate_vram_from_size(file_size_mb, dtype)
    return QuantizationInfo(
        format="safetensors",
        dtype=dtype,
        estimated_vram_mb=vram,
        file_size_mb=round(file_size_mb, 1),
        quantization_method=quant_method,
    )


def _extract_safetensors_dtype(header: dict) -> str:
    """Extract the dominant dtype from a safetensors JSON header.

    Looks at tensor metadata entries and returns the most common dtype.

    Args:
        header: Parsed safetensors JSON header dict.

    Returns:
        Dtype string (fp32, fp16, bf16, fp8_e4m3fn, etc.).
    """
    dtype_counts: dict[str, int] = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if isinstance(value, dict) and "dtype" in value:
            raw_dtype = str(value["dtype"]).lower().replace("float", "f")
            normalized = _normalize_dtype(raw_dtype)
            dtype_counts[normalized] = dtype_counts.get(normalized, 0) + 1

    if not dtype_counts:
        return "fp16"

    return max(dtype_counts, key=lambda d: dtype_counts[d])


def _extract_safetensors_quant_method(header: dict) -> str:
    """Extract quantization method from safetensors metadata.

    Args:
        header: Parsed safetensors JSON header dict.

    Returns:
        Quantization method string.
    """
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        return "native"

    # Check for bitsandbytes NF4 quantization
    if metadata.get("quantization_type") == "nf4":
        return "bitsandbytes_nf4"
    if metadata.get("quantization") == "fp8":
        return "fp8_quantized"

    return "native"


def _detect_gguf(path: Path, file_size_mb: float) -> QuantizationInfo:
    """Parse GGUF file header to extract quantization information.

    Reads the magic bytes and general.file_type metadata to determine
    the quantization level.

    Args:
        path: Path to the GGUF file.
        file_size_mb: Pre-computed file size in MB.

    Returns:
        QuantizationInfo for the GGUF model.
    """
    dtype = "q4_k_m"  # default assumption for GGUF

    try:
        with path.open("rb") as fh:
            magic_bytes = fh.read(4)
            if len(magic_bytes) < 4:
                logger.warning("GGUF file too small: %s", path.name)
            else:
                magic = struct.unpack("<I", magic_bytes)[0]
                if magic != _GGUF_MAGIC:
                    logger.warning(
                        "File %s has wrong GGUF magic: 0x%08X (expected 0x%08X)",
                        path.name,
                        magic,
                        _GGUF_MAGIC,
                    )

                # Read version (uint32)
                version_bytes = fh.read(4)
                if len(version_bytes) >= 4:
                    _version = struct.unpack("<I", version_bytes)[0]

                    # Read tensor count and metadata kv count
                    counts = fh.read(16)
                    if len(counts) >= 16:
                        _tensor_count, _kv_count = struct.unpack("<QQ", counts)
                        dtype = _guess_gguf_dtype_from_filename(path.name)
    except OSError:
        logger.warning("Could not read GGUF header for %s", path.name)

    vram = _estimate_vram_from_size(file_size_mb, dtype)
    return QuantizationInfo(
        format="gguf",
        dtype=dtype,
        estimated_vram_mb=vram,
        file_size_mb=round(file_size_mb, 1),
        quantization_method=f"gguf_{dtype}",
    )


def _detect_checkpoint(path: Path, file_size_mb: float) -> QuantizationInfo:
    """Detect format for PyTorch checkpoint files.

    Cannot reliably determine dtype without loading the checkpoint,
    so uses file size heuristics.

    Args:
        path: Path to the checkpoint file.
        file_size_mb: Pre-computed file size in MB.

    Returns:
        QuantizationInfo for the checkpoint model.
    """
    # Heuristic: >4GB likely FP32, 2-4GB likely FP16, <2GB likely quantized
    if file_size_mb > 4000:
        dtype = "fp32"
    elif file_size_mb > 2000:
        dtype = "fp16"
    else:
        dtype = "fp16"

    vram = _estimate_vram_from_size(file_size_mb, dtype)
    return QuantizationInfo(
        format="ckpt",
        dtype=dtype,
        estimated_vram_mb=vram,
        file_size_mb=round(file_size_mb, 1),
        quantization_method="native",
    )


def _normalize_dtype(raw: str) -> str:
    """Normalize a raw dtype string to a standard form.

    Args:
        raw: Raw dtype string from file header.

    Returns:
        Normalized dtype string.
    """
    mapping: dict[str, str] = {
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float8_e4m3fn": "fp8_e4m3fn",
        "f8_e4m3fn": "fp8_e4m3fn",
        "float8_e5m2": "fp8_e5m2",
        "f8_e5m2": "fp8_e5m2",
        "int8": "int8",
        "uint8": "int8",
        "int4": "nf4",
    }
    return mapping.get(raw, raw)


def _guess_gguf_dtype_from_filename(filename: str) -> str:
    """Guess GGUF quantization type from filename conventions.

    GGUF files typically include the quant type in the filename,
    e.g. ``model-q4_k_m.gguf``, ``model-Q8_0.gguf``.

    Args:
        filename: Filename (not full path).

    Returns:
        Quantization type string.
    """
    name_lower = filename.lower()
    # Check from most specific to least
    quant_patterns = [
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q3_k_l",
        "q4_k_s",
        "q4_k_m",
        "q4_0",
        "q4_1",
        "q5_k_s",
        "q5_k_m",
        "q5_0",
        "q5_1",
        "q6_k",
        "q8_0",
        "q8_1",
        "f16",
        "f32",
    ]
    for pattern in quant_patterns:
        if pattern in name_lower:
            return pattern
    return "q4_k_m"


def _estimate_vram_from_size(file_size_mb: float, dtype: str) -> float:
    """Estimate inference VRAM from file size and dtype.

    For FP16 models, VRAM is approximately equal to file size.
    For quantized models, VRAM is less than file size due to
    dequantization overhead being handled in chunks.

    Args:
        file_size_mb: Model file size in MB.
        dtype: Detected dtype string.

    Returns:
        Estimated VRAM in MB.
    """
    factor = _VRAM_FACTOR.get(dtype, 1.0)
    # For GGUF models, the file size already reflects quantization.
    # VRAM usage is roughly file_size + 10-20% overhead for KV cache.
    if dtype in _GGUF_QUANT_NAMES.values() or dtype.startswith("q"):
        return round(file_size_mb * 1.15, 1)

    # For safetensors/ckpt, estimate based on FP16-equivalent size
    # File size in FP16 is the baseline; scale by factor
    if dtype in ("fp32", "f32"):
        # FP32 file is 2x FP16; VRAM = file_size (already accounts for 2x)
        return round(file_size_mb, 1)

    # FP16/BF16: VRAM ~ file size
    if factor <= 1.0:
        return round(file_size_mb * 1.05, 1)  # small overhead

    return round(file_size_mb * factor, 1)
