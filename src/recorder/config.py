from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class SegmenterConfig:
    aggressiveness: int = 2           # 0~3
    padding_ms: int = 300             # pre/post buffer
    silence_ms: int = 600             # end after this much silence
    max_utterance_ms: int = 15000     # safety cap
    trigger_ratio: float = 0.6
    on_speech_start: Optional[Callable[[], None]] = None


@dataclass
class RecorderConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    device: Optional[int] = None
    latency: str = "low"
    enable_segmenter: bool = True
    chunk_sec: float = 4.0            # used when enable_segmenter=False
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
