from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Tuple

import numpy as np
import webrtcvad

from src.recorder.config import SegmenterConfig


class VADSegmenter:
    def __init__(self, cfg: SegmenterConfig, sample_rate: int, frame_ms: int):
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10/20/30 for WebRTC VAD")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be 8000/16000/32000/48000 for WebRTC VAD")
        if not (0 <= cfg.aggressiveness <= 3):
            raise ValueError("aggressiveness must be 0..3")

        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self.on_speech_start = cfg.on_speech_start

        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # int16 => 2 bytes/sample

        self.padding_frames = int(cfg.padding_ms / frame_ms)
        self.silence_frames_to_end = int(cfg.silence_ms / frame_ms)
        self.max_frames = int(cfg.max_utterance_ms / frame_ms)

    @staticmethod
    def _float32_to_int16_bytes(x: np.ndarray) -> bytes:
        x = np.clip(x, -1.0, 1.0)
        y = (x * 32767.0).astype(np.int16)
        return y.tobytes()

    def segment(self, frames: Iterator[np.ndarray]) -> np.ndarray:
        ring: Deque[Tuple[bytes, bool]] = deque(maxlen=self.padding_frames)

        triggered = False
        voiced_frames: List[bytes] = []
        silence_count = 0
        total_frames = 0

        for frame_f32 in frames:
            frame_b = self._float32_to_int16_bytes(frame_f32)
            if len(frame_b) != self.frame_bytes:
                continue

            is_speech = self.vad.is_speech(frame_b, self.sample_rate)
            total_frames += 1

            if not triggered:
                ring.append((frame_b, is_speech))
                num_voiced = sum(1 for _, s in ring if s)
                if num_voiced > self.cfg.trigger_ratio * ring.maxlen:
                    triggered = True
                    if self.on_speech_start is not None:
                        try:
                            self.on_speech_start()
                        except Exception:
                            pass
                    voiced_frames.extend(b for b, _ in ring)
                    ring.clear()
                    silence_count = 0
            else:
                voiced_frames.append(frame_b)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1

                if silence_count >= self.silence_frames_to_end or total_frames >= self.max_frames:
                    break

        audio_i16 = np.frombuffer(b"".join(voiced_frames), dtype=np.int16)
        return audio_i16.astype(np.float32) / 32767.0
