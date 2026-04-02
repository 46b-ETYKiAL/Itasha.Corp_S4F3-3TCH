"""ComfyUI server lifecycle management.

Provides start, stop, restart, and health checking for a local
ComfyUI server process. Supports both comfy-cli and direct
subprocess invocation with ``shell=False`` always.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEALTH_ENDPOINT = "/system_stats"
_INTERRUPT_ENDPOINT = "/interrupt"
_DEFAULT_STARTUP_TIMEOUT = 60.0
_POLL_INTERVAL = 1.0


@dataclasses.dataclass
class ServerConfig:
    """Configuration for a ComfyUI server instance.

    Attributes:
        host: Listen address.
        port: Listen port.
        cuda_device: CUDA device index (None for default).
        force_fp16: Force FP16 inference.
        preview_method: Preview method (auto, taesd, latent2rgb, none).
        extra_args: Additional CLI arguments.
    """

    host: str = "127.0.0.1"
    port: int = 8188
    cuda_device: int | None = None
    force_fp16: bool = False
    preview_method: str = "auto"
    extra_args: list[str] = dataclasses.field(default_factory=list)

    def to_args(self) -> list[str]:
        """Convert config to CLI argument list.

        Returns:
            List of CLI argument strings.
        """
        args = [
            "--listen",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.cuda_device is not None:
            args.extend(["--cuda-device", str(self.cuda_device)])
        if self.force_fp16:
            args.append("--force-fp16")
        if self.preview_method != "auto":
            args.extend(["--preview-method", self.preview_method])
        args.extend(self.extra_args)
        return args


class ServerLifecycle:
    """Manages the lifecycle of a ComfyUI server process.

    Detects whether ``comfy-cli`` is available and uses it when
    possible. Falls back to direct ``python main.py`` subprocess.
    Always uses ``subprocess`` with ``shell=False``.

    Args:
        comfyui_path: Path to ComfyUI installation directory.
            If None, relies on comfy-cli being on PATH.
    """

    def __init__(self, comfyui_path: str | None = None) -> None:
        self._comfyui_path: Path | None = Path(comfyui_path) if comfyui_path else None
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._config: ServerConfig = ServerConfig()
        self._comfy_cli: str | None = shutil.which("comfy")

    @property
    def has_comfy_cli(self) -> bool:
        """Whether comfy-cli is available on PATH."""
        return self._comfy_cli is not None

    async def start(
        self,
        config: ServerConfig | None = None,
        *,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
    ) -> bool:
        """Start the ComfyUI server.

        If the server is already running, returns True immediately.
        Waits up to ``startup_timeout`` seconds for the health check
        to pass before returning.

        Args:
            config: Server configuration (uses defaults if None).
            startup_timeout: Max seconds to wait for server readiness.

        Returns:
            True if the server is running and healthy.
        """
        if await self.is_running():
            logger.info("ComfyUI server already running.")
            return True

        self._config = config or ServerConfig()
        cmd = self._build_start_command(self._config)

        logger.info("Starting ComfyUI: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self._comfyui_path) if self._comfyui_path else None,
        )

        return await self._wait_for_ready(startup_timeout)

    async def stop(self, timeout: float = 10.0) -> bool:
        """Stop the ComfyUI server gracefully.

        First attempts to interrupt via the API, then sends SIGTERM,
        then waits for the process to exit.

        Args:
            timeout: Max seconds to wait for graceful shutdown.

        Returns:
            True if the server was stopped.
        """
        if not await self.is_running():
            logger.info("ComfyUI server is not running.")
            return True

        # Try API interrupt first
        await self._send_interrupt()

        if self._process is not None:
            try:
                self._process.terminate()
                start = time.monotonic()
                while time.monotonic() - start < timeout:
                    if self._process.poll() is not None:
                        break
                    await asyncio.sleep(0.5)
                else:
                    logger.warning("Graceful stop timed out, killing process.")
                    self._process.kill()
                    self._process.wait(timeout=5)
            except OSError as exc:
                logger.warning("Error stopping process: %s", exc)

        self._process = None
        logger.info("ComfyUI server stopped.")
        return True

    async def restart(
        self,
        config: ServerConfig | None = None,
        *,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
    ) -> bool:
        """Restart the server, optionally with new config.

        Args:
            config: New config (reuses previous if None).
            startup_timeout: Max seconds to wait for readiness.

        Returns:
            True if restart succeeded and server is healthy.
        """
        await self.stop()
        return await self.start(
            config or self._config,
            startup_timeout=startup_timeout,
        )

    async def is_running(self) -> bool:
        """Check if the server process is alive and responsive.

        Returns:
            True if the process is alive and the health endpoint responds.
        """
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            return False

        if self._process is None:
            return await self._check_external_server()

        try:
            health = await self.health_check()
            return health.get("status") == "ok"
        except Exception:
            return False

    async def health_check(self) -> dict[str, Any]:
        """Query the ComfyUI health/system_stats endpoint.

        Returns:
            Dict with server status info, or error details.
        """
        import httpx  # lazy import for optional dep

        url = f"http://{self._config.host}:{self._config.port}{_HEALTH_ENDPOINT}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                data["status"] = "ok"
                return data
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def get_pid(self) -> int | None:
        """Get the PID of the managed server process.

        Returns:
            Process ID if running, None otherwise.
        """
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_start_command(self, config: ServerConfig) -> list[str]:
        """Build the command list to start ComfyUI.

        Uses comfy-cli if available, otherwise direct python invocation.
        """
        if self._comfy_cli is not None:
            return [self._comfy_cli, "launch", "--", *config.to_args()]

        if self._comfyui_path is not None:
            main_py = self._comfyui_path / "main.py"
            return ["python", str(main_py), *config.to_args()]

        return ["python", "-m", "comfy", "launch", "--", *config.to_args()]

    async def _wait_for_ready(self, timeout: float) -> bool:
        """Poll the health endpoint until ready or timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._process is not None and self._process.poll() is not None:
                logger.error("ComfyUI process exited during startup.")
                return False
            try:
                health = await self.health_check()
                if health.get("status") == "ok":
                    logger.info("ComfyUI server ready.")
                    return True
            except Exception:
                pass
            await asyncio.sleep(_POLL_INTERVAL)

        logger.error("ComfyUI server did not become ready within %ss.", timeout)
        return False

    async def _send_interrupt(self) -> None:
        """Send an interrupt request to the ComfyUI API."""
        import httpx  # lazy import

        url = f"http://{self._config.host}:{self._config.port}{_INTERRUPT_ENDPOINT}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url)
        except Exception:
            pass  # Server may already be shutting down

    async def _check_external_server(self) -> bool:
        """Check if an external ComfyUI server is running."""
        try:
            health = await self.health_check()
            return health.get("status") == "ok"
        except Exception:
            return False
