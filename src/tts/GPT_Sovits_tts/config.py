from __future__ import annotations

from dataclasses import dataclass, field
import os


def _env_bool(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes", "y")


@dataclass
class GPTSovitsRemoteConfig:
    backend: str = "gpt_sovits_remote"
    model: str = "gpt-sovits"
    endpoint: str = field(default_factory=lambda: os.environ.get("GPT_SOVITS_ENDPOINT", "https://127.0.0.1:9450/tts"))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("GPT_SOVITS_TIMEOUT_S", "120")))
    verify_ssl: bool = field(default_factory=lambda: _env_bool("GPT_SOVITS_VERIFY_SSL", "0"))
    ca_cert_file: str | None = field(default_factory=lambda: os.environ.get("GPT_SOVITS_CA_CERT_FILE"))

    # Default inference fields expected by GPT-SoVITS api_v2 /tts
    text_lang: str = field(default_factory=lambda: os.environ.get("GPT_SOVITS_TEXT_LANG", "zh"))
    prompt_lang: str = field(default_factory=lambda: os.environ.get("GPT_SOVITS_PROMPT_LANG", "zh"))
    ref_audio_path: str | None = field(default_factory=lambda: os.environ.get("GPT_SOVITS_REF_AUDIO_PATH"))
    prompt_text: str = field(default_factory=lambda: os.environ.get("GPT_SOVITS_PROMPT_TEXT", ""))
    text_split_method: str = field(default_factory=lambda: os.environ.get("GPT_SOVITS_TEXT_SPLIT_METHOD", "cut5"))
    batch_size: int = field(default_factory=lambda: int(os.environ.get("GPT_SOVITS_BATCH_SIZE", "1")))
    speed_factor: float = field(default_factory=lambda: float(os.environ.get("GPT_SOVITS_SPEED_FACTOR", "1.0")))
