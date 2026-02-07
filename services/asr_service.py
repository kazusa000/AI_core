from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.asr.base import ASRResult
from src.asr.factory import ASR_REGISTRY, create_asr

from services.common import build_config, load_wav_upload
from services.runtime import ensure_remote_backend_ready

app = FastAPI(title="ai_core ASR Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "asr"}


@app.post("/v1/asr/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="WAV file"),
    backend: str = Form("paraformer"),
    sample_rate: int = Form(16000),
    config_json: str | None = Form(None),
) -> dict:
    name = backend.strip().lower()
    entry = ASR_REGISTRY.get(name)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown ASR backend: {name}")

    try:
        cfg_dict = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"config_json must be valid JSON: {exc}") from exc

    cfg = build_config(entry.cfg_cls, cfg_dict)

    if entry.runtime_type == "remote_managed":
        endpoint = getattr(cfg, "endpoint", None)
        verify_ssl = bool(getattr(cfg, "verify_ssl", False))
        if not endpoint:
            raise HTTPException(status_code=500, detail=f"Remote backend '{name}' missing endpoint config")
        try:
            ensure_remote_backend_ready(
                service_type="asr",
                model_name=entry.model_name,
                endpoint=endpoint,
                verify_ssl=verify_ssl,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Remote backend startup failed: {exc}") from exc

    asr = create_asr(name, cfg)

    wav, sr = await load_wav_upload(audio)
    use_sr = int(sample_rate or sr)

    res: ASRResult = asr.transcribe(wav, sample_rate=use_sr)
    return {
        "text": (res.text or "").strip(),
        "lang": res.lang,
        "backend": res.backend,
        "sample_rate": use_sr,
    }
