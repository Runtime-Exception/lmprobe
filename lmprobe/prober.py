"""Model availability prober."""

import asyncio
import time
import json
import logging
from typing import AsyncGenerator
from dataclasses import dataclass, field

import httpx

from .models import Endpoint, AvailableModel, ProbeResult

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Thread-safe registry of available models."""
    _models: dict[str, list[AvailableModel]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_model(self, model: AvailableModel) -> None:
        async with self._lock:
            if model.model_id not in self._models:
                self._models[model.model_id] = []
            # Check if this exact endpoint/key combo already exists
            for existing in self._models[model.model_id]:
                if (existing.endpoint_url == model.endpoint_url and
                    existing.api_key == model.api_key):
                    # Update existing
                    existing.last_checked = model.last_checked
                    existing.is_available = model.is_available
                    existing.latency_ms = model.latency_ms
                    return
            self._models[model.model_id].append(model)

    async def remove_model(self, model_id: str, endpoint_url: str, api_key: str) -> None:
        async with self._lock:
            if model_id in self._models:
                self._models[model_id] = [
                    m for m in self._models[model_id]
                    if not (m.endpoint_url == endpoint_url and m.api_key == api_key)
                ]

    async def get_provider(self, model_id: str) -> AvailableModel | None:
        """Get the best available provider for a model (lowest latency)."""
        async with self._lock:
            if model_id not in self._models:
                return None
            available = [m for m in self._models[model_id] if m.is_available]
            if not available:
                return None
            # Sort by latency, None values last
            available.sort(key=lambda m: m.latency_ms if m.latency_ms else float('inf'))
            return available[0]

    async def get_all_models(self) -> list[str]:
        """Get list of all available model IDs."""
        async with self._lock:
            return [
                model_id for model_id, providers in self._models.items()
                if any(p.is_available for p in providers)
            ]

    async def get_model_info(self, model_id: str) -> list[AvailableModel]:
        """Get all providers for a model."""
        async with self._lock:
            return list(self._models.get(model_id, []))


class Prober:
    """Probes endpoints to check model availability."""

    def __init__(
        self,
        endpoints: list[Endpoint],
        registry: ModelRegistry,
        timeout: float = 30.0,
        max_concurrent: int = 20
    ):
        self.endpoints = endpoints
        self.registry = registry
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._client: httpx.AsyncClient | None = None
        self._probe_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        self._stop_event.set()
        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def probe_model(
        self,
        model_id: str,
        endpoint_url: str,
        api_key: str
    ) -> ProbeResult:
        """Probe a single model at a specific endpoint."""
        client = await self._get_client()
        key_prefix = api_key[:8] + "..."

        # Try OpenAI-compatible endpoint first
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Use a minimal request to test availability
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "stream": False
        }

        start_time = time.time()

        try:
            # Try /v1/chat/completions
            url = f"{endpoint_url}/v1/chat/completions"
            response = await client.post(url, headers=headers, json=payload)

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return ProbeResult(
                    model_id=model_id,
                    endpoint_url=endpoint_url,
                    api_key_prefix=key_prefix,
                    success=True,
                    latency_ms=latency_ms
                )
            elif response.status_code == 404:
                # Try without /v1 prefix
                url = f"{endpoint_url}/chat/completions"
                response = await client.post(url, headers=headers, json=payload)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return ProbeResult(
                        model_id=model_id,
                        endpoint_url=endpoint_url,
                        api_key_prefix=key_prefix,
                        success=True,
                        latency_ms=latency_ms
                    )

            # Parse error message
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", str(response.status_code))
            except Exception:
                error_msg = f"HTTP {response.status_code}"

            return ProbeResult(
                model_id=model_id,
                endpoint_url=endpoint_url,
                api_key_prefix=key_prefix,
                success=False,
                error=error_msg
            )

        except httpx.TimeoutException:
            return ProbeResult(
                model_id=model_id,
                endpoint_url=endpoint_url,
                api_key_prefix=key_prefix,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            return ProbeResult(
                model_id=model_id,
                endpoint_url=endpoint_url,
                api_key_prefix=key_prefix,
                success=False,
                error=str(e)
            )

    async def probe_endpoint(self, endpoint: Endpoint) -> AsyncGenerator[ProbeResult, None]:
        """Probe all models at a specific endpoint."""
        semaphore = asyncio.Semaphore(5)  # Limit concurrent probes per endpoint

        async def probe_with_limit(model_id: str, api_key: str) -> ProbeResult:
            async with semaphore:
                return await self.probe_model(model_id, endpoint.base_url, api_key)

        # Only test first key for each model to save time
        tasks = []
        for model_id in endpoint.claimed_models[:100]:  # Limit to first 100 models
            if endpoint.api_keys:
                task = probe_with_limit(model_id, endpoint.api_keys[0])
                tasks.append(task)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    async def probe_all(
        self,
        progress_callback: callable | None = None
    ) -> list[ProbeResult]:
        """Probe all endpoints and models."""
        results: list[ProbeResult] = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        total_probes = sum(
            min(len(e.claimed_models), 100) for e in self.endpoints
        )
        completed = 0

        async def bounded_probe(model_id: str, endpoint: Endpoint) -> ProbeResult:
            nonlocal completed
            async with semaphore:
                if endpoint.api_keys:
                    result = await self.probe_model(
                        model_id, endpoint.base_url, endpoint.api_keys[0]
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_probes, result)
                    return result
                return ProbeResult(
                    model_id=model_id,
                    endpoint_url=endpoint.base_url,
                    api_key_prefix="none",
                    success=False,
                    error="No API key"
                )

        tasks = []
        for endpoint in self.endpoints:
            for model_id in endpoint.claimed_models[:100]:
                tasks.append(bounded_probe(model_id, endpoint))

        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)

            # Update registry
            if result.success:
                # Find the full API key
                for endpoint in self.endpoints:
                    if endpoint.base_url == result.endpoint_url:
                        for key in endpoint.api_keys:
                            if key.startswith(result.api_key_prefix.rstrip("...")):
                                await self.registry.add_model(AvailableModel(
                                    model_id=result.model_id,
                                    endpoint_url=result.endpoint_url,
                                    api_key=key,
                                    last_checked=time.time(),
                                    is_available=True,
                                    latency_ms=result.latency_ms
                                ))
                                break
                        break

        return results

    async def start_background_probing(self, interval: float = 300.0) -> None:
        """Start background probing loop."""
        async def probe_loop():
            while not self._stop_event.is_set():
                logger.info("Starting background probe...")
                try:
                    await self.probe_all()
                except Exception as e:
                    logger.error(f"Background probe error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=interval
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Time for next probe

        self._probe_task = asyncio.create_task(probe_loop())
