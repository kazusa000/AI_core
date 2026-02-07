from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from src.recorder import Recorder, RecorderConfig, SegmenterConfig

app = FastAPI(title="ai_core Recorder Service", version="1.0.0")


class RecorderRequest(BaseModel):
    sample_rate: int = 16000
    frame_ms: int = 20
    device: int | None = None
    latency: str = "low"
    enable_segmenter: bool = True
    chunk_sec: float = 4.0

    # Segmenter options when enable_segmenter=True
    aggressiveness: int = 2
    padding_ms: int = 300
    silence_ms: int = 600
    max_utterance_ms: int = 15000
    trigger_ratio: float = 0.6


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "recorder"}


@app.post("/v1/recorder/capture")
def capture(req: RecorderRequest) -> Response:
    seg_cfg = SegmenterConfig(
        aggressiveness=req.aggressiveness,
        padding_ms=req.padding_ms,
        silence_ms=req.silence_ms,
        max_utterance_ms=req.max_utterance_ms,
        trigger_ratio=req.trigger_ratio,
    )
    rec_cfg = RecorderConfig(
        sample_rate=req.sample_rate,
        frame_ms=req.frame_ms,
        device=req.device,
        latency=req.latency,
        enable_segmenter=req.enable_segmenter,
        chunk_sec=req.chunk_sec,
        segmenter=seg_cfg,
    )

    recorder = Recorder(rec_cfg)
    wav = recorder.listen()

    if wav is None or len(wav) == 0:
        return Response(status_code=204)

    wav = np.asarray(wav, dtype=np.float32)
    duration_s = float(len(wav)) / float(req.sample_rate)

    buf = io.BytesIO()
    sf.write(buf, wav, req.sample_rate, format="WAV", subtype="PCM_16")
    payload = buf.getvalue()

    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(req.sample_rate),
            "X-Duration-S": f"{duration_s:.3f}",
        },
    )
