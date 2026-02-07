from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParaformerConfig:
    model: str = "paraformer-zh"  # 
    device: str = "cuda:0"     # "cuda:0" / "cpu"
    hub: str = "hf"            # "hf" or "ms"
    use_punc: bool = False
    use_vad: bool = False
    hotword: Optional[str] = None
    batch_size_s: int = 0
