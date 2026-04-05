"""Batch generation orchestrator for ComfyUI.

Provides parameter sweeping, seed sweeping, and concurrent queue
management for bulk image generation via the ComfyUI API.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BatchConfig:
    """Configuration for a batch generation run.

    Attributes:
        base_prompt: The base positive prompt text.
        base_params: Default generation parameters (cfg, steps, etc.).
        sweep_params: Parameters to sweep — name maps to list of values.
        seed_count: Number of sequential seeds per combination.
        max_concurrent: Maximum concurrent queue depth.
        output_dir: Directory for organising results.
    """

    base_prompt: str
    base_params: dict[str, Any]
    sweep_params: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    seed_count: int = 1
    max_concurrent: int = 5
    output_dir: str = ""


@dataclasses.dataclass
class BatchResult:
    """Aggregated results of a batch generation run.

    Attributes:
        total: Total number of items in the batch.
        completed: Number of successfully completed items.
        failed: Number of failed items.
        results: Per-item result dicts with params and status.
        elapsed_seconds: Wall-clock duration of the batch.
    """

    total: int
    completed: int
    failed: int
    results: list[dict[str, Any]]
    elapsed_seconds: float = 0.0


class BatchGenerator:
    """Orchestrates batch image generation against a ComfyUI server.

    Generates all parameter combinations from a BatchConfig, then
    queues them with concurrency control and progress tracking.

    Args:
        comfyui_url: Base URL of the ComfyUI API server.
    """

    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188") -> None:
        self._url = comfyui_url.rstrip("/")

    @property
    def comfyui_url(self) -> str:
        """Return the configured ComfyUI server URL."""
        return self._url

    def generate_combinations(self, config: BatchConfig) -> list[dict[str, Any]]:
        """Expand sweep_params into all parameter combinations.

        Each combination is a copy of base_params with the swept
        values overridden. Seeds are appended as sequential values
        starting from the base_params seed (default 0).

        Args:
            config: The batch configuration.

        Returns:
            List of parameter dicts, one per generation job.
        """
        sweep_keys = sorted(config.sweep_params.keys())
        sweep_values = [config.sweep_params[k] for k in sweep_keys]

        if not sweep_values:
            combos = [{}]
        else:
            combos = [
                dict(zip(sweep_keys, vals)) for vals in itertools.product(*sweep_values)
            ]

        base_seed = config.base_params.get("seed", 0)
        all_jobs: list[dict[str, Any]] = []

        for combo in combos:
            for seed_offset in range(config.seed_count):
                params = {**config.base_params, **combo}
                params["seed"] = base_seed + seed_offset
                params["prompt"] = config.base_prompt
                all_jobs.append(params)

        return all_jobs

    def estimate_batch_size(self, config: BatchConfig) -> int:
        """Estimate the total number of jobs in a batch.

        Args:
            config: The batch configuration.

        Returns:
            Number of generation jobs that would be created.
        """
        sweep_keys = sorted(config.sweep_params.keys())
        sweep_values = [config.sweep_params[k] for k in sweep_keys]

        if not sweep_values:
            combo_count = 1
        else:
            combo_count = 1
            for vals in sweep_values:
                combo_count *= len(vals)

        return combo_count * max(config.seed_count, 1)

    async def run_batch(
        self,
        config: BatchConfig,
        *,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> BatchResult:
        """Execute a full batch generation run.

        Queues all combinations against the ComfyUI API with
        concurrency limited to config.max_concurrent.

        Args:
            config: Batch configuration.
            progress_callback: Called with (completed, total, latest_result).

        Returns:
            Aggregated BatchResult.
        """
        jobs = self.generate_combinations(config)
        total = len(jobs)

        if not jobs:
            return BatchResult(total=0, completed=0, failed=0, results=[])

        output_dir = Path(config.output_dir) if config.output_dir else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max(config.max_concurrent, 1))
        results: list[dict[str, Any]] = []
        completed = 0
        failed = 0
        start_time = time.monotonic()

        async def _run_one(index: int, params: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed, failed
            async with semaphore:
                result = await self._queue_prompt(params, index, output_dir)
                if result.get("status") == "success":
                    completed += 1
                else:
                    failed += 1
                if progress_callback is not None:
                    progress_callback(completed + failed, total, result)
                return result

        tasks = [_run_one(i, params) for i, params in enumerate(jobs)]
        results = await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start_time
        return BatchResult(
            total=total,
            completed=completed,
            failed=failed,
            results=list(results),
            elapsed_seconds=round(elapsed, 2),
        )

    async def _queue_prompt(
        self,
        params: dict[str, Any],
        index: int,
        output_dir: Path | None,
    ) -> dict[str, Any]:
        """Queue a single prompt to the ComfyUI API.

        Args:
            params: Generation parameters.
            index: Job index in the batch.
            output_dir: Optional directory for saving result metadata.

        Returns:
            Result dict with status, params, and any error info.
        """
        import httpx  # lazy import for optional dep

        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": self._build_workflow_payload(params),
            "client_id": prompt_id,
        }

        result: dict[str, Any] = {
            "index": index,
            "params": params,
            "prompt_id": prompt_id,
        }

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self._url}/prompt",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                result["status"] = "success"
                result["response"] = data
        except Exception as exc:
            logger.warning("Batch job %d failed: %s", index, exc)
            result["status"] = "failed"
            result["error"] = str(exc)

        if output_dir is not None:
            meta_path = output_dir / f"job_{index:04d}.json"
            meta_path.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8",
            )

        return result

    @staticmethod
    def _build_workflow_payload(params: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal ComfyUI workflow payload from parameters.

        This creates a basic txt2img workflow structure. For complex
        workflows, use TemplateManager to render a full template and
        pass it through base_params['workflow'].

        Args:
            params: Generation parameters.

        Returns:
            ComfyUI-compatible workflow dict.
        """
        if "workflow" in params:
            return params["workflow"]

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": params.get("seed", 0),
                    "steps": params.get("steps", 20),
                    "cfg": params.get("cfg", 7.0),
                    "sampler_name": params.get("sampler", "euler"),
                    "scheduler": params.get("scheduler", "normal"),
                    "denoise": params.get("denoise", 1.0),
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": params.get("prompt", ""),
                },
            },
        }
