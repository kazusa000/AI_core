# src/tts/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Type

from src.tts.base import BaseTTS
from src.tts.Genie_tts import GenieTTSConfig, GenieTTS
from src.tts.GPT_Sovits_tts import GPTSovitsRemoteConfig, GPTSovitsRemoteTTS

RuntimeType = Literal["local", "remote_managed"]
TTSName = Literal["genie_tts", "gpt_sovits_remote"]


@dataclass(frozen=True)
class TTSBackendEntry:
    cfg_cls: Type[object]
    model_cls: Type[BaseTTS]
    model_name: str
    model_dir: str
    runtime_type: RuntimeType = "local"


TTS_REGISTRY: Dict[str, TTSBackendEntry] = {
    "genie_tts": TTSBackendEntry(
        cfg_cls=GenieTTSConfig,
        model_cls=GenieTTS,
        model_name="genie_tts",
        model_dir="src/tts/Genie_tts",
        runtime_type="local",
    ),
    "gpt_sovits_remote": TTSBackendEntry(
        cfg_cls=GPTSovitsRemoteConfig,
        model_cls=GPTSovitsRemoteTTS,
        model_name="gpt_sovits_remote",
        model_dir="src/tts/GPT_Sovits_tts",
        runtime_type="remote_managed",
    ),
}


def create_tts(name: TTSName, cfg: Optional[object] = None) -> BaseTTS:
    """
    返回一个具有 .synthesize(text, ...)->TTSResult 的对象
    """
    norm_name = name.lower().strip()
    entry = TTS_REGISTRY.get(norm_name)
    if entry is None:
        raise ValueError(f"Unknown TTS backend: {name}")
    return entry.model_cls(cfg or entry.cfg_cls())
