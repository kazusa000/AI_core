from __future__ import annotations

from dataclasses import fields, is_dataclass
import io
from typing import Any, Dict, Type

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

WAV_MEDIA_TYPES = {"audio/wav", "audio/x-wav", "application/octet-stream"}


def _filter_dataclass_kwargs(cfg_cls: Type[object], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not is_dataclass(cfg_cls):
        return cfg
    names = {f.name for f in fields(cfg_cls)}
    unknown = sorted(set(cfg) - names)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config keys for {cfg_cls.__name__}: {', '.join(unknown)}",
        )
    return {k: v for k, v in cfg.items() if k in names}


def build_config(cfg_cls: Type[object], cfg: Dict[str, Any] | None) -> object:
    cfg = cfg or {}
    kwargs = _filter_dataclass_kwargs(cfg_cls, cfg)
    try:
        return cfg_cls(**kwargs)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config for {cfg_cls.__name__}: {exc}") from exc


def validate_upload_content_type(upload: UploadFile) -> None:
    if upload.content_type and upload.content_type.lower() not in WAV_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{upload.content_type}'. "
                "Please upload WAV audio (audio/wav)."
            ),
        )


async def load_wav_upload(upload: UploadFile) -> tuple[np.ndarray, int]:
    validate_upload_content_type(upload)
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded WAV is empty")

    try:
        data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode WAV: {exc}") from exc

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim != 1:
        raise HTTPException(status_code=400, detail="Audio must be mono or stereo WAV")

    return audio, int(sample_rate)
