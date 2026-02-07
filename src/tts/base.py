# src/tts/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import threading


# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    audio_format: Optional[str] = None  # e.g. "wav", "mp3"
    backend: Optional[str] = None
    model: Optional[str] = None


class TTSConfigBase(Protocol):
    backend: str
    model: str


# =========================
# Cancel / Interrupt
# =========================

class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancelledError()


class CancelledError(RuntimeError):
    pass


# =========================
# TTS Interface
# =========================

class BaseTTS(Protocol):
    """
    后端需要实现 synthesize(text, ...) -> TTSResult
    """

    cfg: TTSConfigBase

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        sample_rate: Optional[int] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> TTSResult:
        """
        必须实现：
        - 返回合成后的音频字节
        - 需要在合适的地方检查 cancel_token.is_cancelled()
        """
        ...
