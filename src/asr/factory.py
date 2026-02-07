# src/asr/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Type

from src.asr.base import ASRBackend
from src.asr.whisper import FasterWhisperASR, ASRConfig
from src.asr.paraformer import ParaformerASR, ParaformerConfig

RuntimeType = Literal["local", "remote_managed"]
ASRName = Literal["whisper", "paraformer"]


@dataclass(frozen=True)
class ASRBackendEntry:
    cfg_cls: Type[object]
    model_cls: Type[ASRBackend]
    model_name: str
    model_dir: str
    runtime_type: RuntimeType = "local"


ASR_REGISTRY: Dict[str, ASRBackendEntry] = {
    "whisper": ASRBackendEntry(
        cfg_cls=ASRConfig,
        model_cls=FasterWhisperASR,
        model_name="whisper",
        model_dir="src/asr/whisper",
        runtime_type="local",
    ),
    "paraformer": ASRBackendEntry(
        cfg_cls=ParaformerConfig,
        model_cls=ParaformerASR,
        model_name="paraformer",
        model_dir="src/asr/paraformer",
        runtime_type="local",
    ),
}


def create_asr(name: ASRName, cfg: Optional[object] = None) -> ASRBackend:
    """
    返回一个具有 .transcribe(audio, sample_rate=16000)->ASRResult 的对象
    """
    norm_name = name.lower().strip()
    entry = ASR_REGISTRY.get(norm_name)
    if entry is None:
        raise ValueError(f"Unknown ASR backend: {name}")
    return entry.model_cls(cfg or entry.cfg_cls())
