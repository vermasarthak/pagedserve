"""Pydantic schemas for the PagedServe HTTP API (OpenAI-compatible subset)."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = "pagedserve"
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "pagedserve"
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    stream: bool = False
    stop: Optional[List[str]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
