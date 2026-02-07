from __future__ import annotations

from typing import Iterator

import numpy as np
import sounddevice as sd

from src.recorder.config import RecorderConfig

class AudioStreamRecorder:
    """
    Capture microphone audio as fixed-size frames (float32 mono).
    """
    def __init__(self, cfg: RecorderConfig = RecorderConfig()):
        self.cfg = cfg
        self.frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)

    def frame_generator(self) -> Iterator[np.ndarray]:
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.cfg.device,
            latency=self.cfg.latency,
        ) as stream:
            while True:
                data, _overflowed = stream.read(self.frame_samples)
                yield data[:, 0]
