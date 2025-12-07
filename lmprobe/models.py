"""Pydantic models for API requests and responses."""

from typing import Any, Literal
from pydantic import BaseModel, Field


# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "unknown"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


# Claude-compatible models
class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ClaudeChatRequest(BaseModel):
    model: str
    messages: list[ClaudeMessage]
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    system: str | None = None


class ClaudeContentBlock(BaseModel):
    type: str = "text"
    text: str


class ClaudeUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ClaudeChatResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ClaudeContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


# Internal models
class Endpoint(BaseModel):
    base_url: str
    api_keys: list[str]
    claimed_models: list[str]


class AvailableModel(BaseModel):
    model_id: str
    endpoint_url: str
    api_key: str
    last_checked: float
    is_available: bool = True
    latency_ms: float | None = None


class ProbeResult(BaseModel):
    model_id: str
    endpoint_url: str
    api_key_prefix: str  # First 8 chars for identification
    success: bool
    latency_ms: float | None = None
    error: str | None = None
