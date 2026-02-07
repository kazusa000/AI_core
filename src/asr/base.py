# src/asr/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union, Optional
from pathlib import Path
import numpy as np

AudioInput = Union[str, Path, np.ndarray]

@dataclass
class ASRResult:
    text: str
    lang: Optional[str] = None
    backend: Optional[str] = None

class ASRBackend(Protocol):
    """
    ASR 后端接口：任何模型只要实现 transcribe() 就能接入 always_listen。
    """
    def transcribe(self, audio: AudioInput, sample_rate: int = 16000) -> ASRResult: ...
