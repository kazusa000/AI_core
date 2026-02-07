# src/llm/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Protocol, Literal
import threading


# =========================
# Data structures
# =========================

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: dict


@dataclass(frozen=True)
class MessagePart:
    type: Literal["text", "tool_call", "tool_result"]
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    parts: List[MessagePart]


class LLMConfigBase(Protocol):
    backend: str               # "gemini" / "openai" / ...
    model: str


@dataclass
class LLMResponse:
    text: str
    usage: Optional[dict] = None
    backend: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class LLMChunk:
    """
    流式输出单元：通常是 delta（增量文本）。
    - text_delta: 本次新增文本（推荐）
    - is_final: 是否结束（最后一个 chunk）
    """
    text_delta: str = ""
    is_final: bool = False


# =========================
# Cancel / Interrupt
# =========================

class CancelToken:
    """
    任何时候都可以 cancel()；stream/generate 需要在合适的地方检查 is_cancelled。
    """
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancelledError()


class CancelledError(RuntimeError):
    pass


# =========================
# LLM Interface
# =========================

class BaseLLM(Protocol):
    """
    后端需要实现 stream()；generate() 默认用 stream 拼出来。
    """

    cfg: LLMConfigBase

    def stream(self, messages: List[LLMMessage], cancel_token: Optional[CancelToken] = None) -> Iterator[LLMChunk]:
        """
        必须实现：
        - 边生成边 yield chunk
        - 需要频繁检查 cancel_token.is_cancelled() 并尽快停止
        """
        ...

    def generate(self, messages: List[LLMMessage], cancel_token: Optional[CancelToken] = None) -> LLMResponse:
        """
        默认实现：把 stream() 的 delta 拼起来。
        后端一般不必覆写（除非要更精确的 usage 统计等）。
        """
        text_parts: List[str] = []
        backend = getattr(self.cfg, "backend", None)
        model = getattr(self.cfg, "model", None)

        for ch in self.stream(messages, cancel_token=cancel_token):
            if cancel_token is not None and cancel_token.is_cancelled():
                # 允许更快退出（即便后端忘了检查）
                raise CancelledError()
            if ch.text_delta:
                text_parts.append(ch.text_delta)
            if ch.is_final:
                break

        return LLMResponse(text="".join(text_parts), backend=backend, model=model)


# =========================
# Helper: non-streaming fallback wrapper
# =========================

class NonStreamingAdapter:
    """
    如果某个后端暂时只会 blocking 的 generate_once()，
    可以用这个适配器临时提供 stream()：
    - 缺点：中间无法“优雅打断”，只能在结束时丢弃结果
    - 用于过渡期
    """
    def __init__(self, cfg: LLMConfigBase):
        self.cfg = cfg

    def generate_once(self, messages: List[LLMMessage]) -> LLMResponse:
        raise NotImplementedError

    def stream(self, messages: List[LLMMessage], cancel_token: Optional[CancelToken] = None) -> Iterator[LLMChunk]:
        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancelledError()
        res = self.generate_once(messages)
        if cancel_token is not None and cancel_token.is_cancelled():
            # 已经晚了，但至少不输出
            raise CancelledError()
        yield LLMChunk(text_delta=res.text, is_final=True)
