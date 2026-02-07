# src/asr/paraformer/model.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile

import numpy as np
import soundfile as sf
from funasr import AutoModel

from src.asr.base import ASRResult, AudioInput
from src.asr.paraformer.config import ParaformerConfig


class ParaformerASR:
    def __init__(self, cfg: ParaformerConfig = ParaformerConfig()):
        self.cfg = cfg

        kwargs = {
            "model": cfg.model,
            "device": cfg.device,
            "hub": cfg.hub,
            "disable_update": True,
        }

        # 你已有 webrtcvad 切句，通常不要再开 funasr 自带 vad/punc
        if cfg.use_vad:
            kwargs["vad_model"] = "fsmn-vad"
        if cfg.use_punc:
            kwargs["punc_model"] = "ct-punc"

        self.model = AutoModel(**kwargs)

    @staticmethod
    def _parse_text(res: Any) -> str:
        if isinstance(res, list) and res:
            first = res[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip()
        return str(res).strip()

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
        """
        hotword = self.cfg.hotword or ""

        # 1) 路径输入：最稳
        if isinstance(audio, (str, Path)):
            res = self.model.generate(
                input=str(audio),
                batch_size_s=self.cfg.batch_size_s,
                hotword=hotword,
            )
            text = self._parse_text(res)
            return ASRResult(
                text=text,
                lang=None,
                backend=f"paraformer/{self.cfg.model}:{self.cfg.device}",
            )

        # 2) ndarray 输入：先直喂，失败落盘 wav
        x = self._ensure_float32_mono(audio)

        try:
            res = self.model.generate(
                input=x,
                batch_size_s=self.cfg.batch_size_s,
                hotword=hotword,
            )
            text = self._parse_text(res)
            return ASRResult(
                text=text,
                lang=None,
                backend=f"paraformer/{self.cfg.model}:{self.cfg.device}",
            )
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, x, samplerate=sample_rate, subtype="PCM_16")
                res = self.model.generate(
                    input=f.name,
                    batch_size_s=self.cfg.batch_size_s,
                    hotword=hotword,
                )
                text = self._parse_text(res)
                return ASRResult(
                    text=text,
                    lang=None,
                    backend=f"paraformer/{self.cfg.model}:{self.cfg.device}(tmpwav)",
                )


if __name__ == "__main__":
    # 自测：
    # python -m src.asr.paraformer.model out/test.wav
    import sys

    wav = sys.argv[1] if len(sys.argv) > 1 else "out/test.wav"
    asr = ParaformerASR(ParaformerConfig(model="paraformer-zh", device="cuda:0", hub="hf"))
    res = asr.transcribe(wav)
    print(res.text)
    print(f"[{res.backend}]")
