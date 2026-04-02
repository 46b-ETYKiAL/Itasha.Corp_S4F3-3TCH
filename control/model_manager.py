"""Model management for ComfyUI checkpoints and LoRAs.

Provides model listing with metadata, architecture detection from
safetensors headers, precision conversion, weighted-sum merging,
and checksum verification. All write operations create new files —
originals are never modified in-place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = frozenset({"safetensors", "ckpt", "gguf", "pt", "bin"})
_SAFETENSORS_EXTENSIONS = frozenset({".safetensors"})

_ARCHITECTURE_KEYS: dict[str, list[str]] = {
    "flux": ["double_blocks", "img_in", "txt_in"],
    "sd3": ["joint_blocks", "context_embedder"],
    "sdxl": ["conditioner.embedders.1", "input_blocks.7"],
    "sd15": ["cond_stage_model", "model.diffusion_model"],
}


@dataclasses.dataclass
class ModelInfo:
    """Metadata about a model file.

    Attributes:
        name: Model filename (without directory).
        path: Absolute path to the model file.
        size_bytes: File size in bytes.
        format: File format (safetensors, ckpt, gguf, etc.).
        architecture: Detected architecture (sd15, sdxl, flux, sd3, unknown).
        dtype: Detected data type (fp16, fp32, fp8, bf16, unknown).
    """

    name: str
    path: str
    size_bytes: int
    format: str
    architecture: str
    dtype: str


class ModelManager:
    """Manages ComfyUI model files.

    Handles listing, metadata inspection, architecture detection,
    precision conversion, and weighted-sum merging. All mutating
    operations produce new output files.

    Args:
        models_dir: Root directory containing model files.
    """

    def __init__(self, models_dir: str | Path) -> None:
        self._dir = Path(models_dir)
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        """Return the models root directory."""
        return self._dir

    def list_models(self) -> list[ModelInfo]:
        """List all recognised model files with metadata.

        Returns:
            List of ModelInfo for each model file found.
        """
        models: list[ModelInfo] = []
        for path in sorted(self._dir.rglob("*")):
            if not path.is_file():
                continue
            fmt = self._detect_format(path)
            if fmt is None:
                continue
            arch = self._detect_architecture_from_path(path)
            dtype = self._detect_dtype_from_path(path)
            models.append(
                ModelInfo(
                    name=path.name,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    format=fmt,
                    architecture=arch,
                    dtype=dtype,
                )
            )
        return models

    def get_model_info(self, name: str) -> ModelInfo | None:
        """Get info for a specific model by filename.

        Args:
            name: Model filename to look up.

        Returns:
            ModelInfo if found, None otherwise.
        """
        for model in self.list_models():
            if model.name == name:
                return model
        return None

    async def download_model(
        self,
        url: str,
        filename: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Download a model file via HTTP streaming.

        Uses httpx for async streaming download with optional progress
        reporting. The file is written to a temporary name first, then
        renamed on success.

        Args:
            url: URL to download from.
            filename: Target filename within models_dir.
            progress_callback: Called with (bytes_downloaded, total_bytes).

        Returns:
            Absolute path of the downloaded file.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
            ImportError: If httpx is not installed.
        """
        import httpx

        dest = self._dir / filename
        tmp = dest.with_suffix(dest.suffix + ".part")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                with tmp.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, total)

        tmp.rename(dest)
        logger.info("Downloaded model '%s' (%d bytes)", filename, downloaded)
        return str(dest)

    def detect_architecture(self, model_path: str) -> str:
        """Detect model architecture from safetensors header keys.

        Args:
            model_path: Path to the model file.

        Returns:
            Architecture string: sd15, sdxl, flux, sd3, or unknown.
        """
        return self._detect_architecture_from_path(Path(model_path))

    def convert_precision(
        self,
        model_path: str,
        target_dtype: str = "fp16",
        *,
        output_name: str = "",
    ) -> str:
        """Convert model precision by rewriting safetensors metadata.

        Creates a new file with the target dtype annotation in the
        header metadata. Actual tensor data conversion requires a
        full tensor library; this method updates the metadata marker
        so downstream tools respect the intent.

        Args:
            model_path: Path to source model.
            target_dtype: Target precision (fp16, fp32, bf16).
            output_name: Output filename; auto-generated if empty.

        Returns:
            Path to the new converted file.

        Raises:
            ValueError: If the source is not safetensors format.
        """
        src = Path(model_path)
        if src.suffix != ".safetensors":
            msg = "Precision conversion only supports safetensors format."
            raise ValueError(msg)

        header, header_size = self._read_safetensors_header(src)
        if header is None:
            msg = f"Could not read safetensors header from {model_path}"
            raise ValueError(msg)

        metadata = header.get("__metadata__", {})
        metadata["target_dtype"] = target_dtype
        header["__metadata__"] = metadata

        if not output_name:
            output_name = f"{src.stem}_{target_dtype}{src.suffix}"
        dest = self._dir / output_name

        self._write_safetensors_with_header(src, dest, header, header_size)
        logger.info("Converted '%s' → '%s' (target: %s)", src.name, dest.name, target_dtype)
        return str(dest)

    def merge_models(
        self,
        model_a: str,
        model_b: str,
        alpha: float = 0.5,
        output_name: str = "",
    ) -> str:
        """Record a weighted-sum merge configuration for two models.

        Creates a merge manifest (JSON) describing the merge parameters.
        Actual tensor-level merging requires a full tensor library;
        this method produces the configuration for external tooling
        (e.g., sd-meh, mergekit) to execute.

        Args:
            model_a: Path to first model.
            model_b: Path to second model.
            alpha: Interpolation weight (0.0 = all A, 1.0 = all B).
            output_name: Output manifest name; auto-generated if empty.

        Returns:
            Path to the merge manifest JSON.

        Raises:
            ValueError: If alpha is out of [0, 1] range.
            FileNotFoundError: If either model file is missing.
        """
        if not 0.0 <= alpha <= 1.0:
            msg = f"Alpha must be in [0, 1], got {alpha}"
            raise ValueError(msg)

        path_a, path_b = Path(model_a), Path(model_b)
        for p in (path_a, path_b):
            if not p.exists():
                msg = f"Model not found: {p}"
                raise FileNotFoundError(msg)

        if not output_name:
            output_name = f"merge_{path_a.stem}_{path_b.stem}_a{alpha:.2f}"

        manifest = {
            "type": "weighted_sum",
            "model_a": str(path_a),
            "model_b": str(path_b),
            "alpha": alpha,
            "output_name": output_name,
            "checksum_a": self.verify_checksum(model_a),
            "checksum_b": self.verify_checksum(model_b),
        }

        dest = self._dir / f"{output_name}.merge.json"
        dest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Created merge manifest '%s'", dest.name)
        return str(dest)

    def verify_checksum(self, model_path: str) -> str:
        """Compute SHA-256 checksum of a model file.

        Args:
            model_path: Path to the model file.

        Returns:
            Hex-encoded SHA-256 digest.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(model_path)
        if not path.exists():
            msg = f"File not found: {model_path}"
            raise FileNotFoundError(msg)

        sha = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(path: Path) -> str | None:
        """Detect model format from file extension."""
        ext = path.suffix.lstrip(".").lower()
        if ext in _SUPPORTED_FORMATS:
            return ext
        return None

    def _detect_architecture_from_path(self, path: Path) -> str:
        """Detect architecture from a model file path."""
        if path.suffix not in _SAFETENSORS_EXTENSIONS:
            return "unknown"
        header, _ = self._read_safetensors_header(path)
        if header is None:
            return "unknown"
        return self._match_architecture(header)

    def _detect_dtype_from_path(self, path: Path) -> str:
        """Detect primary dtype from safetensors header."""
        if path.suffix not in _SAFETENSORS_EXTENSIONS:
            return "unknown"
        header, _ = self._read_safetensors_header(path)
        if header is None:
            return "unknown"
        return self._extract_dtype(header)

    @staticmethod
    def _read_safetensors_header(path: Path) -> tuple[dict[str, Any] | None, int]:
        """Read the JSON header from a safetensors file.

        Returns:
            Tuple of (header_dict, header_byte_size) or (None, 0).
        """
        try:
            with path.open("rb") as f:
                size_bytes = f.read(8)
                if len(size_bytes) < 8:
                    return None, 0
                header_size = struct.unpack("<Q", size_bytes)[0]
                if header_size > 100_000_000:  # sanity cap at 100MB
                    return None, 0
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes)
                return header, header_size
        except (OSError, json.JSONDecodeError, struct.error):
            return None, 0

    @staticmethod
    def _write_safetensors_with_header(src: Path, dest: Path, header: dict[str, Any], old_header_size: int) -> None:
        """Rewrite a safetensors file with a modified header."""
        new_header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
        new_header_size = len(new_header_bytes)

        with src.open("rb") as fin, dest.open("wb") as fout:
            fout.write(struct.pack("<Q", new_header_size))
            fout.write(new_header_bytes)
            # Skip past old header
            fin.seek(8 + old_header_size)
            while True:
                chunk = fin.read(65536)
                if not chunk:
                    break
                fout.write(chunk)

    @staticmethod
    def _match_architecture(header: dict[str, Any]) -> str:
        """Match architecture based on header key patterns."""
        keys_str = " ".join(header.keys())
        for arch, markers in _ARCHITECTURE_KEYS.items():
            if any(marker in keys_str for marker in markers):
                return arch
        return "unknown"

    @staticmethod
    def _extract_dtype(header: dict[str, Any]) -> str:
        """Extract the dominant dtype from tensor descriptors."""
        dtype_counts: dict[str, int] = {}
        for key, value in header.items():
            if key.startswith("__"):
                continue
            if isinstance(value, dict) and "dtype" in value:
                dt = value["dtype"].lower()
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        if not dtype_counts:
            return "unknown"
        return max(dtype_counts, key=dtype_counts.get)  # type: ignore[arg-type]
