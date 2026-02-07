# src/asr/whisper/model.py
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from src.asr.base import ASRResult, AudioInput
from src.asr.whisper.config import ASRConfig


class FasterWhisperASR:
    def __init__(self, cfg: ASRConfig = ASRConfig()):
        self.cfg = cfg

        device = cfg.device
        compute_type = cfg.compute_type

        # auto 策略：优先 cuda + float16；否则 cpu + int8
        if device == "auto":
            device = "cuda"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(
            cfg.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    @staticmethod
    def _ensure_float32_mono(audio: np.ndarray) -> np.ndarray:
        a = np.asarray(audio)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1:
            raise ValueError("audio must be mono 1D array (shape=(n_samples,))")
        return a.astype(np.float32, copy=False)

    def transcribe(self, audio: AudioInput, sample_rate: int = 16000) -> ASRResult:
        """
        audio: wav路径(str/Path) 或 numpy float32 mono (N,)
        sample_rate: 保留用于接口统一（whisper 通常不需要）
        """
        if isinstance(audio, np.ndarray):
            audio_in = self._ensure_float32_mono(audio)
        else:
            audio_in = str(audio)

        segments_iter, info = self.model.transcribe(
            audio_in,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
        )

        text = "".join(s.text for s in segments_iter).strip()
        lang = getattr(info, "language", None)

        return ASRResult(
            text=text,
            lang=lang,
            backend=f"whisper/{self.cfg.model_size}:{self.device},{self.compute_type}",
        )


if __name__ == "__main__":
    # 自测：
    # python -m src.asr.whisper.model out/test.wav
    import sys

    wav = sys.argv[1] if len(sys.argv) > 1 else "out/test.wav"
    asr = FasterWhisperASR(ASRConfig(model_size="small", device="auto", compute_type="auto", language=None))
    res = asr.transcribe(wav)
    print(res.text)
    print(f"[lang={res.lang}] [{res.backend}]")
