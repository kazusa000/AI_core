from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRConfig:
    model_size: str = "small"
    device: str = "auto"              # "auto" / "cpu" / "cuda"
    compute_type: str = "auto"        # "auto" / "float16" / "int8" / "float32"
    language: Optional[str] = "zh"    # None=自动识别
    beam_size: int = 5
    vad_filter: bool = False          # 你已有 webrtcvad 切句，通常 False
