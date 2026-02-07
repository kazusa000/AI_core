from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os


def _env_bool(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes", "y")


@dataclass
class GenieTTSConfig:
    backend: str = "genie_tts"
    data_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("GENIE_DATA_DIR", "src/tts/Genie_tts/GenieData/GenieData")
    )
    character_name: Optional[str] = field(default_factory=lambda: os.environ.get("GENIE_CHARACTER", "feibi"))
    onnx_model_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "GENIE_ONNX_DIR",
            "src/tts/Genie_tts/CharacterModels/CharacterModels/v2ProPlus/feibi/tts_models",
        )
    )
    language: Optional[str] = field(default_factory=lambda: os.environ.get("GENIE_LANGUAGE", "zh"))
    voice_dir: str = field(
        default_factory=lambda: os.environ.get(
            "GENIE_VOICE_DIR",
            "src/tts/Genie_tts/CharacterModels/CharacterModels",
        )
    )
    voice_profile: Optional[str] = field(default_factory=lambda: os.environ.get("GENIE_VOICE_PROFILE"))
    reference_audio: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "GENIE_REF_AUDIO",
            "src/tts/Genie_tts/CharacterModels/CharacterModels/v2ProPlus/feibi/prompt_wav/"
            "zh_vo_Main_Linaxita_2_1_10_26.wav",
        )
    )
    reference_text: Optional[str] = field(default_factory=lambda: os.environ.get("GENIE_REF_TEXT"))
    reference_text_path: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "GENIE_REF_TEXT_PATH",
            "src/tts/Genie_tts/CharacterModels/CharacterModels/v2ProPlus/feibi/prompt_wav.json",
        )
    )
    output_dir: str = field(default_factory=lambda: os.environ.get("GENIE_OUTPUT_DIR", "out"))
    keep_output: bool = field(default_factory=lambda: _env_bool("GENIE_KEEP_OUTPUT", "0"))
