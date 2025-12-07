"""FastAPI server with OpenAI and Claude-compatible endpoints."""

import time
import uuid
import json
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    OpenAIChatRequest, OpenAIChatResponse, OpenAIChoice, OpenAIMessage,
    OpenAIUsage, OpenAIModel, OpenAIModelList,
    ClaudeChatRequest, ClaudeChatResponse, ClaudeContentBlock, ClaudeUsage,
    AvailableModel
)
from .prober import ModelRegistry, Prober
from .config import parse_api_config_ex

logger = logging.getLogger(__name__)


class LMProbeServer:
    """Main server class managing the API proxy."""

    def __init__(
        self,
        config_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        probe_interval: float = 300.0,
        force_probe: bool = False
    ):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.probe_interval = probe_interval
        self.force_probe = force_probe
        self.registry = ModelRegistry()
        self.prober: Prober | None = None
        self.endpoints = []
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def _stream_openai_response(
        self,
        client: httpx.AsyncClient,
        provider: AvailableModel,
        headers: dict,
        payload: dict
    ) -> AsyncGenerator[bytes, None]:
        """Stream OpenAI-compatible response."""
        try:
            async with client.stream(
                "POST",
                f"{provider.endpoint_url}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    # Try without /v1
                    async with client.stream(
                        "POST",
                        f"{provider.endpoint_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response2:
                        async for chunk in response2.aiter_bytes():
                            yield chunk
                else:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    async def _stream_claude_response(
        self,
        client: httpx.AsyncClient,
        provider: AvailableModel,
        headers: dict,
        payload: dict
    ) -> AsyncGenerator[bytes, None]:
        """Stream Claude-compatible response by converting from OpenAI stream."""
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Send message_start event
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': payload['model']}})}\n\n".encode()

        # Send content_block_start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()

        try:
            async with client.stream(
                "POST",
                f"{provider.endpoint_url}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n".encode()
                        except json.JSONDecodeError:
                            pass

            # Send content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

            # Send message_delta
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n".encode()

            # Send message_stop
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n".encode()

    def create_app(self) -> FastAPI:
        server = self  # Capture self for use in route handlers

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info(f"Loading config from {server.config_path}")
            config = parse_api_config_ex(server.config_path)
            server.endpoints = config.endpoints
            logger.info(f"Loaded {len(server.endpoints)} endpoints")

            server.prober = Prober(
                server.endpoints,
                server.registry,
                max_concurrent=30
            )

            # Check if config is pre-probed
            if config.is_probed and not server.force_probe:
                # Skip probing - directly register all models as available
                logger.info("Config is pre-probed, skipping initial probe")
                if config.probed_timestamp:
                    from datetime import datetime
                    probe_time = datetime.fromtimestamp(config.probed_timestamp)
                    logger.info(f"Config was probed at: {probe_time}")

                # Register all models from pre-probed config
                for endpoint in server.endpoints:
                    for model in endpoint.claimed_models:
                        for key in endpoint.api_keys[:1]:  # Use first key
                            await server.registry.add_model(AvailableModel(
                                model_id=model,
                                endpoint_url=endpoint.base_url,
                                api_key=key,
                                last_checked=config.probed_timestamp or time.time(),
                                is_available=True,
                                latency_ms=None
                            ))

                models = await server.registry.get_all_models()
                logger.info(f"Registered {len(models)} pre-verified models")

                # Start background probing (will update availability over time)
                await server.prober.start_background_probing(server.probe_interval)
            else:
                # Do initial probe
                if server.force_probe and config.is_probed:
                    logger.info("Force probe enabled, probing pre-probed config...")
                else:
                    logger.info("Starting initial model probe...")

                results = await server.prober.probe_all(
                    progress_callback=lambda c, t, r: logger.info(
                        f"Probe progress: {c}/{t} - {r.model_id}: {'OK' if r.success else r.error}"
                    ) if c % 50 == 0 else None
                )

                successful = sum(1 for r in results if r.success)
                logger.info(f"Initial probe complete: {successful}/{len(results)} successful")

                # Start background probing
                await server.prober.start_background_probing(server.probe_interval)

            yield

            # Shutdown
            if server.prober:
                await server.prober.close()
            if server._client:
                await server._client.aclose()

        app = FastAPI(
            title="LMProbe",
            description="Model availability detection and API proxy",
            version="0.1.0",
            lifespan=lifespan
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        @app.get("/")
        async def root():
            return {"status": "ok", "service": "lmprobe"}

        @app.get("/health")
        async def health():
            models = await server.registry.get_all_models()
            return {"status": "healthy", "available_models": len(models)}

        # OpenAI-compatible endpoints
        @app.get("/v1/models")
        @app.get("/models")
        async def list_models():
            model_ids = await server.registry.get_all_models()
            return OpenAIModelList(
                data=[
                    OpenAIModel(id=model_id, created=int(time.time()))
                    for model_id in sorted(model_ids)
                ]
            )

        @app.get("/v1/models/{model_id:path}")
        @app.get("/models/{model_id:path}")
        async def get_model(model_id: str):
            providers = await server.registry.get_model_info(model_id)
            if not providers or not any(p.is_available for p in providers):
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            return OpenAIModel(id=model_id, created=int(time.time()))

        @app.post("/v1/chat/completions")
        @app.post("/chat/completions")
        async def openai_chat_completion(request: OpenAIChatRequest):
            provider = await server.registry.get_provider(request.model)
            if not provider:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model} not available"
                )

            client = await server._get_client()
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

            payload = request.model_dump(exclude_none=True)

            if request.stream:
                return StreamingResponse(
                    server._stream_openai_response(client, provider, headers, payload),
                    media_type="text/event-stream"
                )

            # Non-streaming request
            try:
                response = await client.post(
                    f"{provider.endpoint_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    # Try without /v1 prefix
                    response = await client.post(
                        f"{provider.endpoint_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.text
                    )

                return response.json()

            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Request timeout")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Claude-compatible endpoints
        @app.post("/v1/messages")
        @app.post("/messages")
        async def claude_chat_completion(request: ClaudeChatRequest):
            provider = await server.registry.get_provider(request.model)
            if not provider:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model} not available"
                )

            client = await server._get_client()

            # Most providers use OpenAI format even for Claude models
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

            # Convert to OpenAI format
            openai_messages = []
            if request.system:
                openai_messages.append({"role": "system", "content": request.system})
            for msg in request.messages:
                openai_messages.append({"role": msg.role, "content": msg.content})

            openai_payload = {
                "model": request.model,
                "messages": openai_messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            if request.top_p:
                openai_payload["top_p"] = request.top_p
            if request.stop_sequences:
                openai_payload["stop"] = request.stop_sequences

            if request.stream:
                return StreamingResponse(
                    server._stream_claude_response(client, provider, headers, openai_payload),
                    media_type="text/event-stream"
                )

            try:
                response = await client.post(
                    f"{provider.endpoint_url}/v1/chat/completions",
                    headers=headers,
                    json=openai_payload
                )

                if response.status_code != 200:
                    response = await client.post(
                        f"{provider.endpoint_url}/chat/completions",
                        headers=headers,
                        json=openai_payload
                    )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.text
                    )

                # Convert OpenAI response to Claude format
                openai_resp = response.json()

                content_text = ""
                if openai_resp.get("choices"):
                    content_text = openai_resp["choices"][0].get("message", {}).get("content", "")

                usage = openai_resp.get("usage", {})

                return ClaudeChatResponse(
                    id=f"msg_{uuid.uuid4().hex[:24]}",
                    content=[ClaudeContentBlock(text=content_text)],
                    model=request.model,
                    stop_reason=openai_resp.get("choices", [{}])[0].get("finish_reason"),
                    usage=ClaudeUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0)
                    )
                )

            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Request timeout")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Status endpoints
        @app.get("/status")
        async def get_status():
            models = await server.registry.get_all_models()
            return {
                "total_endpoints": len(server.endpoints),
                "available_models": len(models),
                "models": sorted(models)
            }

        @app.get("/status/{model_id:path}")
        async def get_model_status(model_id: str):
            providers = await server.registry.get_model_info(model_id)
            if not providers:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            return {
                "model_id": model_id,
                "providers": [
                    {
                        "endpoint": p.endpoint_url,
                        "available": p.is_available,
                        "latency_ms": p.latency_ms,
                        "last_checked": p.last_checked
                    }
                    for p in providers
                ]
            }

        # Probe endpoints
        @app.post("/probe")
        async def trigger_probe():
            """Manually trigger a probe of all endpoints."""
            if server.prober:
                results = await server.prober.probe_all()
                successful = sum(1 for r in results if r.success)
                return {
                    "probed": len(results),
                    "successful": successful,
                    "failed": len(results) - successful
                }
            raise HTTPException(status_code=500, detail="Prober not initialized")

        return app
