from __future__ import annotations

from itertools import islice

import numpy as np

from src.recorder.config import RecorderConfig
from src.recorder.stream import AudioStreamRecorder
from src.recorder.vad_segmenter import VADSegmenter


class Recorder:
    """
    Microphone recorder with optional speech segmenting.
    """
    def __init__(self, cfg: RecorderConfig = RecorderConfig()):
        self.cfg = cfg
        self.stream = AudioStreamRecorder(cfg)
        self.segmenter = None
        if cfg.enable_segmenter:
            self.segmenter = VADSegmenter(
                cfg.segmenter,
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
            )

    @property
    def sample_rate(self) -> int:
        return self.cfg.sample_rate

    def _read_chunk(self) -> np.ndarray:
        frames = self.stream.frame_generator()
        frames_per_chunk = max(1, int((self.cfg.chunk_sec * 1000) / self.cfg.frame_ms))
        chunk_frames = list(islice(frames, frames_per_chunk))
        if not chunk_frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunk_frames)

    def listen(self) -> np.ndarray:
        if self.segmenter is None:
            return self._read_chunk()
        return self.segmenter.segment(self.stream.frame_generator())


if __name__ == "__main__":
    recorder = Recorder()
    print("Listening... say something, then pause.")
    audio = recorder.listen()
    print(f"Got audio: {len(audio)/recorder.sample_rate:.2f} sec, samples={len(audio)}")
