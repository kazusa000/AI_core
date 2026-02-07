# src/llm/Gemini/model.py
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple
import json

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "GeminiLLM requires package 'google-genai'.\n"
        "Please run:\n"
        "  pip install -U google-genai\n"
        "and make sure you are in the correct virtual environment."
    ) from e

from src.llm.base import LLMMessage, LLMChunk, CancelToken, CancelledError, MessagePart
from src.llm.Gemini.config import GeminiConfig


class GeminiLLM:
    """
    google-genai 后端（Gemini Developer API / Vertex AI）。
    - 实现 stream(messages, cancel_token) -> Iterator[LLMChunk]
    - generate() 由新版 base 的默认实现拼接 stream() 得到
    """

    def __init__(self, cfg: GeminiConfig):
        if not cfg.api_key:
            raise ValueError("GeminiLLM requires cfg.api_key")
        if not cfg.model:
            raise ValueError("GeminiLLM requires cfg.model")

        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.api_key)

    # =========================
    # Public: streaming
    # =========================
    def stream(
        self,
        messages: List[LLMMessage],
        cancel_token: Optional[CancelToken] = None,
        *,
        structured: bool = True,
    ) -> Iterator[LLMChunk]:
        """
        structured=True: 使用 Gemini 的结构化 contents（推荐）
        structured=False: 用旧版 prompt 拼接（fallback）
        """
        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancelledError()

        temperature = self.cfg.temperature
        config = types.GenerateContentConfig(
            temperature=temperature,
            # system_instruction 会在 structured 模式下设置
        )
        if self.cfg.tools:
            config.tools = list(self.cfg.tools)

        if structured:
            system_instruction, contents = self._messages_to_contents(messages)
            if system_instruction:
                config.system_instruction = system_instruction

            # 同步流式接口：generate_content_stream
            accumulated = ""
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.cfg.model,
                    contents=contents,
                    config=config,
                ):
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise CancelledError()

                    piece = getattr(chunk, "text", None) or ""
                    if not piece:
                        continue

                    # 兼容：有的 SDK 返回累计文本，有的返回增量
                    if accumulated and piece.startswith(accumulated):
                        delta = piece[len(accumulated):]
                        accumulated = piece
                    else:
                        delta = piece
                        accumulated += delta

                    if delta:
                        yield LLMChunk(text_delta=delta, is_final=False)
            finally:
                if cancel_token is None or not cancel_token.is_cancelled():
                    yield LLMChunk(text_delta="", is_final=True)
            return

        # fallback：旧式拼 prompt
        prompt = self._messages_to_prompt(messages)
        accumulated = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.cfg.model,
                contents=prompt,
                config=config,
            ):
                if cancel_token is not None and cancel_token.is_cancelled():
                    raise CancelledError()

                piece = getattr(chunk, "text", None) or ""
                if not piece:
                    continue

                if accumulated and piece.startswith(accumulated):
                    delta = piece[len(accumulated):]
                    accumulated = piece
                else:
                    delta = piece
                    accumulated += delta

                if delta:
                    yield LLMChunk(text_delta=delta, is_final=False)
        finally:
            if cancel_token is None or not cancel_token.is_cancelled():
                yield LLMChunk(text_delta="", is_final=True)

    def generate(
        self,
        messages: List[LLMMessage],
        cancel_token: Optional[CancelToken] = None,
    ):
        text_parts: List[str] = []
        backend = getattr(self.cfg, "backend", None)
        model = getattr(self.cfg, "model", None)

        for ch in self.stream(messages, cancel_token=cancel_token):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise CancelledError()
            if ch.text_delta:
                text_parts.append(ch.text_delta)
            if ch.is_final:
                break

        from src.llm.base import LLMResponse

        return LLMResponse(text="".join(text_parts), backend=backend, model=model)

    # =========================
    # Structured contents builder (你要的新增函数)
    # =========================
    @staticmethod
    def _messages_to_contents(messages: List[LLMMessage]) -> Tuple[str, List[types.Content]]:
        """
        把通用 LLMMessage 列表转换为 Gemini 的结构化输入：
        - system: 合并为一个 system_instruction 字符串（放在 GenerateContentConfig.system_instruction）
        - user:   types.UserContent(parts=[Part.from_text(...), ...])
        - assistant: types.ModelContent(parts=[Part.from_text(...), ...])  # Gemini 里 assistant 对应 role='model'
        """
        system_lines: List[str] = []
        contents: List[types.Content] = []

        for m in messages:
            role = (m.role or "").strip().lower()
            parts = GeminiLLM._parts_to_parts(m.parts)
            if not parts:
                continue

            if role == "system":
                system_lines.append(GeminiLLM._parts_to_text(m.parts))
                continue

            if role == "user":
                contents.append(
                    types.UserContent(
                        parts=parts
                    )
                )
                continue

            if role == "assistant":
                contents.append(
                    types.ModelContent(
                        parts=parts
                    )
                )
                continue

            # 兜底：未知 role 当 user
            contents.append(types.UserContent(parts=parts))

        system_instruction = "\n\n".join(system_lines).strip()
        return system_instruction, contents

    # =========================
    # Old prompt builder (fallback)
    # =========================
    @staticmethod
    def _messages_to_prompt(messages: List[LLMMessage]) -> str:
        lines: List[str] = []
        for m in messages:
            text = GeminiLLM._parts_to_text(m.parts)
            if not text:
                continue
            if m.role == "system":
                lines.append(f"[System]\n{text}")
            elif m.role == "user":
                lines.append(f"[User]\n{text}")
            elif m.role == "assistant":
                lines.append(f"[Assistant]\n{text}")
            elif m.role == "tool":
                lines.append(f"[Tool]\n{text}")
        lines.append("[Assistant]\n")
        return "\n\n".join(lines)

    @staticmethod
    def _parts_to_text(parts: List[MessagePart]) -> str:
        lines: List[str] = []
        for part in parts:
            if part.type == "text" and part.text:
                lines.append(part.text)
            elif part.type == "tool_call" and part.tool_call:
                payload = {
                    "type": "tool_call",
                    "name": part.tool_call.name,
                    "arguments": part.tool_call.arguments,
                }
                lines.append(json.dumps(payload, ensure_ascii=False))
            elif part.type == "tool_result" and part.tool_result:
                payload = {
                    "type": "tool_result",
                    "name": part.tool_result.name,
                    "output": part.tool_result.output,
                }
                lines.append(json.dumps(payload, ensure_ascii=False))
        return "\n".join(lines).strip()

    @staticmethod
    def _parts_to_parts(parts: List[MessagePart]) -> List[types.Part]:
        out: List[types.Part] = []
        for part in parts:
            if part.type == "text" and part.text:
                out.append(types.Part.from_text(text=part.text))
                continue

            if part.type == "tool_call" and part.tool_call:
                factory = getattr(types.Part, "from_function_call", None)
                if callable(factory):
                    out.append(
                        factory(
                            name=part.tool_call.name,
                            args=part.tool_call.arguments,
                        )
                    )
                else:
                    payload = {
                        "type": "tool_call",
                        "name": part.tool_call.name,
                        "arguments": part.tool_call.arguments,
                    }
                    out.append(types.Part.from_text(text=json.dumps(payload, ensure_ascii=False)))
                continue

            if part.type == "tool_result" and part.tool_result:
                factory = getattr(types.Part, "from_function_response", None)
                if callable(factory):
                    out.append(
                        factory(
                            name=part.tool_result.name,
                            response=part.tool_result.output,
                        )
                    )
                else:
                    payload = {
                        "type": "tool_result",
                        "name": part.tool_result.name,
                        "output": part.tool_result.output,
                    }
                    out.append(types.Part.from_text(text=json.dumps(payload, ensure_ascii=False)))
                continue
        return out
