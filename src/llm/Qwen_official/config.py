from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class QwenOfficialConfig:
    backend: str = "qwen_official"
    model: str = field(default_factory=lambda: os.environ.get("QWEN_MODEL", "Qwen/Qwen2-7B-Instruct"))
    device: Optional[str] = field(default_factory=lambda: os.environ.get("QWEN_DEVICE"))
    torch_dtype: str = field(default_factory=lambda: os.environ.get("QWEN_TORCH_DTYPE", "auto"))
    trust_remote_code: bool = True
    use_chat_template: bool = True
    device_map: Optional[str] = field(default_factory=lambda: os.environ.get("QWEN_DEVICE_MAP"))
    max_new_tokens: int = field(default_factory=lambda: int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("QWEN_TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.environ.get("QWEN_TOP_P", "0.9")))
    do_sample: bool = field(default_factory=lambda: os.environ.get("QWEN_DO_SAMPLE", "1") != "0")
    repetition_penalty: Optional[float] = field(
        default_factory=lambda: float(os.environ.get("QWEN_REPETITION_PENALTY", "1.0"))
    )
