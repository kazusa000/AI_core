from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Any
import os


@dataclass
class GeminiConfig:
    backend: str = "gemini"
    model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"))
    api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", "AIzaSyCqTdPQ4nl2rnFAqHC3ite5AWmeu1MPoto").strip())
    temperature: float = field(default_factory=lambda: float(os.environ.get("GEMINI_TEMPERATURE", "0.3")))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("GEMINI_TIMEOUT_S", "60")))
    tools: Optional[Sequence[Any]] = None
