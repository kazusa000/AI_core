# src/llm/Qwen_official/model.py
from __future__ import annotations

from typing import Iterator, List, Optional
import json
import threading

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
except ImportError as e:
    raise ImportError(
        "QwenOfficialLLM requires 'transformers' and 'torch'.\n"
        "Please run:\n"
        "  pip install -U transformers torch\n"
        "and make sure you are in the correct virtual environment."
    ) from e

from src.llm.base import (
    LLMMessage,
    LLMChunk,
    LLMResponse,
    CancelToken,
    CancelledError,
    MessagePart,
)
from src.llm.Qwen_official.config import QwenOfficialConfig


class _CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_token: CancelToken):
        self._cancel_token = cancel_token

    def __call__(self, input_ids, scores, **kwargs):
        return self._cancel_token.is_cancelled()


class QwenOfficialLLM:
    """
    Local Qwen-7B Instruct via transformers.
    """

    def __init__(self, cfg: QwenOfficialConfig):
        if not cfg.model:
            raise ValueError("QwenOfficialLLM requires cfg.model")

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_dtype(cfg.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
        )

        device_map = cfg.device_map
        if device_map is None and self.device == "cuda":
            device_map = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=cfg.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        self._generate_lock = threading.Lock()

    def stream(
        self,
        messages: List[LLMMessage],
        cancel_token: Optional[CancelToken] = None,
    ) -> Iterator[LLMChunk]:
        if cancel_token is not None and cancel_token.is_cancelled():
            raise CancelledError()

        with self._generate_lock:
            prompt = self._messages_to_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "do_sample": self.cfg.do_sample,
            }
            if self.cfg.repetition_penalty and self.cfg.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = self.cfg.repetition_penalty

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            stopping = None
            if cancel_token is not None:
                stopping = StoppingCriteriaList([_CancelStoppingCriteria(cancel_token)])

            thread_exc: List[BaseException] = []

            def _run_generate() -> None:
                try:
                    self.model.generate(
                        **inputs,
                        streamer=streamer,
                        stopping_criteria=stopping,
                        **gen_kwargs,
                    )
                except BaseException as exc:
                    thread_exc.append(exc)

            thread = threading.Thread(target=_run_generate, daemon=True)
            thread.start()

            cancelled = False
            try:
                for text in streamer:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        cancelled = True
                        raise CancelledError()
                    if text:
                        yield LLMChunk(text_delta=text, is_final=False)
            finally:
                thread.join()
                if thread_exc and not cancelled:
                    raise RuntimeError("QwenOfficialLLM generate failed") from thread_exc[0]
                if not cancelled:
                    yield LLMChunk(text_delta="", is_final=True)

    def generate(
        self,
        messages: List[LLMMessage],
        cancel_token: Optional[CancelToken] = None,
    ) -> LLMResponse:
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

        return LLMResponse(text="".join(text_parts), backend=backend, model=model)

    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        items: List[dict] = []
        for msg in messages:
            role = (msg.role or "").strip().lower()
            text = self._parts_to_text(msg.parts)
            if not text:
                continue

            if role not in {"system", "user", "assistant"}:
                role = "user"

            items.append({"role": role, "content": text})

        if self.cfg.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                items,
                tokenize=False,
                add_generation_prompt=True,
            )

        lines: List[str] = []
        for it in items:
            role = it["role"]
            text = it["content"]
            if role == "system":
                lines.append(f"[System]\n{text}")
            elif role == "user":
                lines.append(f"[User]\n{text}")
            elif role == "assistant":
                lines.append(f"[Assistant]\n{text}")
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
    def _resolve_dtype(dtype: str):
        if not dtype or dtype == "auto":
            if torch.cuda.is_available():
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        dtype = dtype.lower()
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype}")
