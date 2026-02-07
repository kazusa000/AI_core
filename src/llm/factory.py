# src/llm/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Type

from src.llm.base import BaseLLM
from src.llm.Gemini import GeminiConfig, GeminiLLM
from src.llm.Qwen_official import QwenOfficialConfig, QwenOfficialLLM

RuntimeType = Literal["local", "remote_managed"]
LLMName = Literal["gemini", "qwen_official"]


@dataclass(frozen=True)
class LLMBackendEntry:
    cfg_cls: Type[object]
    model_cls: Type[BaseLLM]
    model_name: str
    model_dir: str
    runtime_type: RuntimeType = "local"


LLM_REGISTRY: Dict[str, LLMBackendEntry] = {
    "gemini": LLMBackendEntry(
        cfg_cls=GeminiConfig,
        model_cls=GeminiLLM,
        model_name="gemini",
        model_dir="src/llm/Gemini",
        runtime_type="local",
    ),
    "qwen_official": LLMBackendEntry(
        cfg_cls=QwenOfficialConfig,
        model_cls=QwenOfficialLLM,
        model_name="qwen_official",
        model_dir="src/llm/Qwen_official",
        runtime_type="local",
    ),
}


def create_llm(
    name: LLMName,
    cfg: Optional[object] = None,
) -> BaseLLM:
    """
    返回一个实现了 BaseLLM 接口的对象。
    """
    norm_name = name.lower().strip()

    entry = LLM_REGISTRY.get(norm_name)
    if entry is None:
        raise ValueError(f"Unknown LLM backend: {name}")
    return entry.model_cls(cfg or entry.cfg_cls())
