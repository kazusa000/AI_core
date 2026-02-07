from __future__ import annotations

import io
import json
import ssl
import urllib.error
import urllib.request
import wave
from typing import Optional

from src.tts.base import CancelToken, CancelledError, TTSResult
from src.tts.GPT_Sovits_tts.config import GPTSovitsRemoteConfig


class GPTSovitsRemoteTTS:
    def __init__(self, cfg: GPTSovitsRemoteConfig):
        self.cfg = cfg

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        sample_rate: Optional[int] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> TTSResult:
        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancelledError()
        if not text.strip():
            raise ValueError("text cannot be empty")

        payload = {
            "text": text,
            "text_lang": self.cfg.text_lang,
            "prompt_lang": self.cfg.prompt_lang,
            "prompt_text": self.cfg.prompt_text,
            "text_split_method": self.cfg.text_split_method,
            "batch_size": self.cfg.batch_size,
            "speed_factor": self.cfg.speed_factor,
            "media_type": "wav",
            "streaming_mode": False,
        }
        if self.cfg.ref_audio_path:
            payload["ref_audio_path"] = self.cfg.ref_audio_path

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        ssl_ctx = None
        if self.cfg.endpoint.lower().startswith("https://"):
            if self.cfg.verify_ssl:
                ssl_ctx = ssl.create_default_context(cafile=self.cfg.ca_cert_file)
            else:
                ssl_ctx = ssl._create_unverified_context()

        try:
            resp = urllib.request.urlopen(req, timeout=self.cfg.timeout_s, context=ssl_ctx)
            audio_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GPT-SoVITS HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GPT-SoVITS request failed: {exc}") from exc

        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancelledError()

        detected_sr = self._parse_wav_sample_rate(audio_bytes)
        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate or detected_sr,
            audio_format="wav",
            backend=self.cfg.backend,
            model=self.cfg.model,
        )

    @staticmethod
    def _parse_wav_sample_rate(payload: bytes) -> int:
        with wave.open(io.BytesIO(payload), "rb") as wf:
            return int(wf.getframerate())
