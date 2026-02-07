# src/tts/Genie_tts/model.py
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path
import threading
import time
import wave
import json
import os
import tempfile

from src.tts.base import CancelToken, CancelledError, TTSResult
from src.tts.Genie_tts.config import GenieTTSConfig


class GenieTTS:
    """
    GENIE (GPT-SoVITS lightweight inference) backend.
    """

    def __init__(self, cfg: GenieTTSConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._loaded_character: Optional[Tuple[str, Optional[str]]] = None
        self._prompt_key: Optional[Tuple[str, str]] = None
        self._genie = None

        if cfg.data_dir:
            data_dir = str(Path(cfg.data_dir).expanduser().resolve())
            os.environ["GENIE_DATA_DIR"] = data_dir
        self._genie = self._import_genie()

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

        with self._lock:
            character_name, onnx_dir, language = self._resolve_character(voice)
            self._ensure_character_loaded(character_name, onnx_dir, language)
            ref_audio, ref_text = self._resolve_reference(character_name, voice)
            if ref_audio and ref_text:
                if self._prompt_key != (ref_audio, ref_text):
                    self._genie.set_reference_audio(
                        character_name=character_name,
                        audio_path=ref_audio,
                        audio_text=ref_text,
                    )
                    self._prompt_key = (ref_audio, ref_text)

            if cancel_token is not None and cancel_token.is_cancelled():
                raise CancelledError()

            out_path: Path
            if self.cfg.keep_output:
                out_path = self._build_output_path()
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                out_path = Path(tmp.name)

            try:
                self._genie.tts(
                    character_name=character_name,
                    text=text,
                    play=False,
                    save_path=str(out_path),
                )
                audio_bytes, detected_sr = self._read_wav(out_path)
            finally:
                if not self.cfg.keep_output:
                    try:
                        out_path.unlink()
                    except FileNotFoundError:
                        pass

        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate or detected_sr,
            audio_format="wav",
            backend=self.cfg.backend,
            model=None,
        )

    def _resolve_character(self, voice: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
        if self.cfg.character_name:
            return self.cfg.character_name, self.cfg.onnx_model_dir, self.cfg.language

        profile = voice or self.cfg.voice_profile
        if not profile:
            raise ValueError("Missing character_name or voice_profile for GENIE.")
        onnx_dir = self._resolve_voice_dir(profile) / "onnx"
        language = self.cfg.language
        return profile, str(onnx_dir), language

    def _ensure_character_loaded(
        self,
        character_name: str,
        onnx_dir: Optional[str],
        language: Optional[str],
    ) -> None:
        key = (character_name, onnx_dir)
        if self._loaded_character == key:
            return

        if onnx_dir:
            if not language:
                raise ValueError("GENIE language is required when loading a custom character.")
            self._genie.load_character(
                character_name=character_name,
                onnx_model_dir=str(Path(onnx_dir).expanduser()),
                language=language,
            )
        else:
            self._genie.load_predefined_character(character_name)

        self._loaded_character = key
        self._prompt_key = None

    def _resolve_reference(self, character_name: str, voice: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if self.cfg.reference_audio and (self.cfg.reference_text or self.cfg.reference_text_path):
            ref_audio = self._require_path(self.cfg.reference_audio, "GENIE_REF_AUDIO")
            ref_text = self._resolve_reference_text()
            return ref_audio, ref_text

        profile = voice or self.cfg.voice_profile or character_name
        voice_dir = self._resolve_voice_dir(profile)
        ref_audio = None
        for name in ("ref.wav", "ref.mp3"):
            candidate = voice_dir / name
            if candidate.is_file():
                ref_audio = str(candidate.resolve())
                break
        ref_text_path = voice_dir / "ref.txt"
        ref_text = ref_text_path.read_text(encoding="utf-8").strip() if ref_text_path.is_file() else None
        if ref_audio and ref_text:
            return ref_audio, ref_text
        return None, None

    def _resolve_reference_text(self) -> str:
        if self.cfg.reference_text_path:
            path = Path(self.cfg.reference_text_path).expanduser()
            if path.is_file():
                if path.suffix.lower() == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        default = data.get("Normal") or next(iter(data.values()), {})
                        if isinstance(default, dict):
                            text = default.get("text")
                            if text:
                                return str(text).strip()
                    raise ValueError(f"Unsupported reference json format: {path}")
                return path.read_text(encoding="utf-8").strip()
        if self.cfg.reference_text:
            possible = Path(self.cfg.reference_text).expanduser()
            if possible.is_file():
                return possible.read_text(encoding="utf-8").strip()
            return self.cfg.reference_text.strip()
        raise ValueError("Reference text missing. Set GENIE_REF_TEXT or GENIE_REF_TEXT_PATH.")

    def _resolve_voice_dir(self, profile: str) -> Path:
        base = Path(self.cfg.voice_dir)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent / base
        if profile and (base / profile).is_dir():
            return base / profile
        return base

    def _build_output_path(self) -> Path:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"genie_tts_{int(time.time() * 1000)}.wav"

    @staticmethod
    def _require_path(value: Optional[str], env_key: str) -> str:
        if not value:
            raise ValueError(f"Missing required config. Set {env_key}.")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_key} does not exist: {path}")
        return str(path.resolve())

    @staticmethod
    def _read_wav(path: Path) -> Tuple[bytes, int]:
        data = path.read_bytes()
        with wave.open(str(path), "rb") as wf:
            return data, wf.getframerate()

    @staticmethod
    def _import_genie():
        try:
            import genie_tts as genie
        except ImportError as e:
            raise ImportError(
                "GenieTTS requires 'genie-tts'.\n"
                "Please run:\n"
                "  pip install genie-tts\n"
                "and make sure you are in the correct virtual environment."
            ) from e
        return genie
