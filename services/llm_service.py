from __future__ import annotations

from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.llm.base import LLMMessage, MessagePart
from src.llm.factory import LLM_REGISTRY, create_llm

from services.common import build_config
from services.runtime import ensure_remote_backend_ready

app = FastAPI(title="ai_core LLM Service", version="1.0.0")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str = Field(..., min_length=1)


class LLMRequest(BaseModel):
    backend: str = "qwen_official"
    messages: list[ChatMessage]
    config: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "llm"}


def _prepare_llm(name: str, config: dict[str, Any] | None):
    entry = LLM_REGISTRY.get(name)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown LLM backend: {name}")

    cfg = build_config(entry.cfg_cls, config)

    if entry.runtime_type == "remote_managed":
        endpoint = getattr(cfg, "endpoint", None)
        verify_ssl = bool(getattr(cfg, "verify_ssl", False))
        if not endpoint:
            raise HTTPException(status_code=500, detail=f"Remote backend '{name}' missing endpoint config")
        try:
            ensure_remote_backend_ready(
                service_type="llm",
                model_name=entry.model_name,
                endpoint=endpoint,
                verify_ssl=verify_ssl,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Remote backend startup failed: {exc}") from exc

    return create_llm(name, cfg)


@app.post("/v1/llm/generate")
def generate(req: LLMRequest) -> dict:
    name = req.backend.strip().lower()
    llm = _prepare_llm(name, req.config)

    messages = [
        LLMMessage(role=m.role, parts=[MessagePart(type="text", text=m.content)])
        for m in req.messages
    ]
    res = llm.generate(messages)
    return {
        "text": res.text,
        "backend": res.backend,
        "model": res.model,
        "usage": res.usage,
    }


@app.post("/v1/llm/stream")
def stream(req: LLMRequest) -> StreamingResponse:
    name = req.backend.strip().lower()
    llm = _prepare_llm(name, req.config)
    messages = [
        LLMMessage(role=m.role, parts=[MessagePart(type="text", text=m.content)])
        for m in req.messages
    ]

    def iter_text() -> Iterator[bytes]:
        for chunk in llm.stream(messages):
            if chunk.text_delta:
                yield chunk.text_delta.encode("utf-8")

    return StreamingResponse(iter_text(), media_type="text/plain; charset=utf-8")
