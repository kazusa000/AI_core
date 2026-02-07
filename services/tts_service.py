from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.tts.factory import TTS_REGISTRY, create_tts
from services.common import build_config
from services.runtime import ensure_remote_backend_ready

app = FastAPI(title="ai_core TTS Service", version="1.0.0")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    backend: str = "genie_tts"
    voice: str | None = None
    sample_rate: int | None = None
    config: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "tts"}


@app.post("/v1/tts/synthesize")
def synthesize(req: TTSRequest) -> Response:
    name = req.backend.strip().lower()
    entry = TTS_REGISTRY.get(name)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown TTS backend: {name}")

    cfg = build_config(entry.cfg_cls, req.config)

    if entry.runtime_type == "remote_managed":
        endpoint = getattr(cfg, "endpoint", None)
        verify_ssl = bool(getattr(cfg, "verify_ssl", False))
        if not endpoint:
            raise HTTPException(status_code=500, detail=f"Remote backend '{name}' missing endpoint config")
        try:
            ensure_remote_backend_ready(
                service_type="tts",
                model_name=entry.model_name,
                endpoint=endpoint,
                verify_ssl=verify_ssl,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Remote backend startup failed: {exc}") from exc

    tts = create_tts(name, cfg)

    try:
        result = tts.synthesize(req.text, voice=req.voice, sample_rate=req.sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    audio_format = (result.audio_format or "wav").lower()
    if audio_format != "wav":
        raise HTTPException(
            status_code=500,
            detail=f"TTS backend returned unsupported format '{audio_format}', expected wav",
        )

    headers = {
        "X-Backend": result.backend or name,
        "X-Sample-Rate": str(result.sample_rate),
    }
    if result.model:
        headers["X-Model"] = result.model

    return Response(content=result.audio_bytes, media_type="audio/wav", headers=headers)
