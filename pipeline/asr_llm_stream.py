# pipeline/asr_llm_stream.py
from __future__ import annotations

import threading
import queue
from typing import List

from src.recorder import Recorder, RecorderConfig
from src.asr.factory import create_asr
from src.asr.base import ASRResult

from src.llm.factory import create_llm  # 用新版 factory
from src.llm.base import (
    LLMMessage,
    MessagePart,
    CancelToken,
    CancelledError,
)

class LatestQueue:
    def __init__(self, maxsize: int = 1) -> None:
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=maxsize)

    def push(self, item: str) -> None:
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    return

    def pop(self, timeout: float = 0.2) -> str | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class InterruptController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: CancelToken | None = None

    def new_token(self) -> CancelToken:
        with self._lock:
            if self._token is not None:
                self._token.cancel()
            self._token = CancelToken()
            return self._token

    def cancel(self) -> None:
        with self._lock:
            if self._token is not None:
                self._token.cancel()


def main():
    # =========================
    # 1) ASR
    # =========================
    asr = create_asr("paraformer")  # "whisper" or "paraformer"

    # =========================
    # 2) LLM (Gemini)
    # =========================
    # 只从环境变量读取 key，别写死
    # export GEMINI_API_KEY="..."
    llm = create_llm("qwen_official")  # "gemini" or "qwen_official"

    system_prompt = "你扮演人工智能助手，说话符合角色风格，不要输出 markdown，不要输出多余格式。"

    history: List[LLMMessage] = [
        LLMMessage(role="system", parts=[MessagePart(type="text", text=system_prompt)])
    ]
    MAX_TURNS = 10  # 保留最近 10 轮对话（user+assistant 算一轮）

    # =========================
    # 3) Recorder segmenting
    # =========================
    recorder = Recorder(
        RecorderConfig(
            sample_rate=16000,
            frame_ms=20,
            enable_segmenter=True,
        )
    )

    # =========================
    # 4) 并发：ASR 线程持续产出 user_text
    # =========================
    user_q = LatestQueue[str]()  # 只保留最新一句（打断语义更自然）
    stop_event = threading.Event()

    def trim_history(hist: List[LLMMessage]) -> List[LLMMessage]:
        keep = 1 + 2 * MAX_TURNS
        if len(hist) > keep:
            return [hist[0]] + hist[-(keep - 1) :]
        return hist

    # 共享状态：用于打断当前 LLM stream
    interrupt = InterruptController(CancelToken)

    def asr_listener_loop():
        print("🎧 Always listening...（Ctrl+C 退出）")
        print("说话 → 停顿 → ASR → Gemini（流式）回复；新说话会打断当前生成\n")

        while not stop_event.is_set():
            audio = recorder.listen()
            duration = len(audio) / recorder.sample_rate

            if duration < 0.25:
                continue

            res: ASRResult = asr.transcribe(audio, sample_rate=recorder.sample_rate)
            user_text = (res.text or "").strip()
            if not user_text:
                continue

            lang = res.lang or "-"
            backend = res.backend or asr.__class__.__name__
            print(f"\n[{backend}] [lang={lang}] {user_text}")

            # 一旦有新输入：立刻打断正在生成的 LLM
            interrupt.cancel()
            user_q.push(user_text)

    def llm_loop():
        nonlocal history
        while not stop_event.is_set():
            user_text = user_q.pop(timeout=0.2)
            if user_text is None:
                continue

            # 更新 history
            history.append(LLMMessage(role="user", parts=[MessagePart(type="text", text=user_text)]))
            history = trim_history(history)

            # 新一轮生成：创建新 token（并 cancel 旧 token）
            token = interrupt.new_token()

            # 流式输出
            print("[gemini] ", end="", flush=True)
            assistant_parts: List[str] = []

            try:
                for ch in llm.stream(history, cancel_token=token):
                    if ch.text_delta:
                        print(ch.text_delta, end="", flush=True)
                        assistant_parts.append(ch.text_delta)
                    if ch.is_final:
                        break

                assistant_text = "".join(assistant_parts).strip()
                if assistant_text:
                    print("")  # 换行
                    history.append(
                        LLMMessage(role="assistant", parts=[MessagePart(type="text", text=assistant_text)])
                    )
                    history = trim_history(history)
                else:
                    print("")  # 换行（空输出也结束）

            except CancelledError:
                # 被新 ASR 打断：输出一行提示（你也可以选择不打印）
                print("\n[gemini] (interrupted)")
                # 不把半截回复写入 history（避免污染上下文）

            except Exception as e:
                print(f"\n[gemini] (error) {e}")

    t_asr = threading.Thread(target=asr_listener_loop, daemon=True)
    t_llm = threading.Thread(target=llm_loop, daemon=True)
    t_asr.start()
    t_llm.start()

    try:
        while True:
            t_asr.join(timeout=1.0)
            t_llm.join(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()
        interrupt.cancel()
        print("\n👋 bye")


if __name__ == "__main__":
    main()
