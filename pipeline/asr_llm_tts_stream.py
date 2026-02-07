from __future__ import annotations

import threading
import io
import time
import queue
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from src.recorder import Recorder, RecorderConfig, SegmenterConfig
from src.asr.factory import create_asr
from src.asr.base import ASRResult
from src.llm.factory import create_llm
from src.llm.base import (
    LLMMessage,
    MessagePart,
    CancelToken as LLMCancelToken,
    CancelledError,
)
from src.tts.factory import create_tts
from src.tts.Genie_tts import GenieTTSConfig
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
        self._token: LLMCancelToken | None = None

    def new_token(self) -> LLMCancelToken:
        with self._lock:
            if self._token is not None:
                self._token.cancel()
            self._token = LLMCancelToken()
            return self._token

    def cancel(self) -> None:
        with self._lock:
            if self._token is not None:
                self._token.cancel()


@dataclass
class SegmentState:
    reply_id: int
    seg_idx: int
    frames_left: int
    tts_ms: float
    text_len: int


class PlaybackBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: "deque[np.ndarray]" = deque()
        self._offset = 0
        self._segments: "deque[SegmentState]" = deque()
        self._completed: "deque[SegmentState]" = deque()

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._segments.clear()
            self._completed.clear()
            self._offset = 0

    def push(self, data: np.ndarray, seg: SegmentState) -> None:
        with self._lock:
            self._chunks.append(data)
            self._segments.append(seg)

    def pop(self, frames: int, channels: int) -> Tuple[np.ndarray, int]:
        with self._lock:
            if not self._chunks:
                return np.zeros((frames, channels), dtype=np.float32), 0
            out = np.zeros((frames, channels), dtype=np.float32)
            filled = 0
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                start = self._offset
                available = len(chunk) - start
                take = min(frames - filled, available)
                out[filled : filled + take] = chunk[start : start + take]
                filled += take
                self._offset += take
                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0
            return out, filled

    def consume(self, frames: int) -> None:
        with self._lock:
            remaining = frames
            while remaining > 0 and self._segments:
                seg = self._segments[0]
                if remaining >= seg.frames_left:
                    remaining -= seg.frames_left
                    self._segments.popleft()
                    self._completed.append(seg)
                else:
                    seg.frames_left -= remaining
                    remaining = 0

    def pop_completed(self) -> List[SegmentState]:
        done: List[SegmentState] = []
        with self._lock:
            while self._completed:
                done.append(self._completed.popleft())
        return done


def main() -> None:
    asr_start = time.perf_counter()
    asr = create_asr("paraformer")
    asr_ms = (time.perf_counter() - asr_start) * 1000.0

    llm_start = time.perf_counter()
    llm = create_llm("qwen_official")
    llm_ms = (time.perf_counter() - llm_start) * 1000.0

    tts_start = time.perf_counter()
    tts = create_tts("genie_tts", GenieTTSConfig())
    tts_ms = (time.perf_counter() - tts_start) * 1000.0

    print(f"[load] ASR: {asr_ms:.1f} ms, LLM: {llm_ms:.1f} ms, TTS: {tts_ms:.1f} ms")

    system_prompt = "你扮演人工智能助手，说话符合角色风格，不要输出 markdown，不要输出多余格式，说话内容不要太长。"
    history: List[LLMMessage] = [
        LLMMessage(role="system", parts=[MessagePart(type="text", text=system_prompt)])
    ]
    max_turns = 10

    user_q = LatestQueue()
    stop_event = threading.Event()
    interrupt = InterruptController()
    tts_q: "queue.Queue[Tuple[int, int, int, str]]" = queue.Queue(maxsize=12)
    audio_q: "queue.Queue[Tuple[int, int, int, bytes, int, float, int]]" = queue.Queue(maxsize=12)
    gen_lock = threading.Lock()
    current_gen_id = 0
    response_counter = 0
    playback_reset = threading.Event()

    split_chars = set("，,。！？；.!?;")
    min_chars = 10
    max_chars = 1000
    max_wait_s = 10

    def bump_gen_id() -> int:
        nonlocal current_gen_id
        with gen_lock:
            current_gen_id += 1
            return current_gen_id

    def get_gen_id() -> int:
        with gen_lock:
            return current_gen_id

    def clear_tts_queue() -> None:
        while True:
            try:
                tts_q.get_nowait()
            except queue.Empty:
                return

    def clear_audio_queue() -> None:
        while True:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                return

    def push_tts_segment(gen_id: int, reply_id: int, seg_idx: int, text: str) -> None:
        if not text.strip():
            return
        while True:
            try:
                tts_q.put_nowait((gen_id, reply_id, seg_idx, text))
                return
            except queue.Full:
                try:
                    tts_q.get_nowait()
                except queue.Empty:
                    return

    def on_speech_start() -> None:
        interrupt.cancel()
        sd.stop()
        new_gen_id = bump_gen_id()
        clear_tts_queue()
        clear_audio_queue()
        playback_reset.set()
        print(f"\n[vad] speech start -> interrupt (gen_id={new_gen_id})")

    recorder = Recorder(
        RecorderConfig(
            sample_rate=16000,
            frame_ms=20,
            enable_segmenter=True,
            segmenter=SegmenterConfig(
                aggressiveness=3,
                padding_ms=500,
                silence_ms=800,
                max_utterance_ms=15000,
                on_speech_start=on_speech_start,
            ),
        )
    )

    def trim_history(hist: List[LLMMessage]) -> List[LLMMessage]:
        keep = 1 + 2 * max_turns
        if len(hist) > keep:
            return [hist[0]] + hist[-(keep - 1) :]
        return hist

    def asr_listener_loop() -> None:
        print("🎧 Listening...（Ctrl+C 退出）")
        print("说话 → 停顿 → ASR → LLM → TTS\n")
        while not stop_event.is_set():
            audio = recorder.listen()
            duration = len(audio) / recorder.sample_rate
            if duration < 1:
                continue
            asr_start = time.perf_counter()
            res: ASRResult = asr.transcribe(audio, sample_rate=recorder.sample_rate)
            asr_ms = (time.perf_counter() - asr_start) * 1000.0
            user_text = (res.text or "").strip()
            if not user_text:
                continue
            lang = res.lang or "-"
            backend = res.backend or asr.__class__.__name__
            print(f"\n[{backend}] [lang={lang}] {user_text} (ASR {asr_ms:.1f} ms)")

            interrupt.cancel()
            user_q.push(user_text)

    def llm_tts_loop() -> None:
        nonlocal history
        nonlocal response_counter
        while not stop_event.is_set():
            user_text = user_q.pop(timeout=0.2)
            if user_text is None:
                continue

            history.append(LLMMessage(role="user", parts=[MessagePart(type="text", text=user_text)]))
            history = trim_history(history)

            token = interrupt.new_token()
            gen_id = bump_gen_id()
            response_counter += 1
            reply_id = response_counter
            seg_idx = 0
            assistant_parts: List[str] = []
            print(f"[llm#R{reply_id}] (gen_id={gen_id}) ", end="", flush=True)

            try:
                llm_start = time.perf_counter()
                buffer = ""
                last_emit = time.perf_counter()

                def emit_segment(segment: str) -> None:
                    nonlocal last_emit
                    nonlocal seg_idx
                    seg_idx += 1
                    print(f"\n[llm#R{reply_id}] emit seg {seg_idx} (chars={len(segment)})")
                    push_tts_segment(gen_id, reply_id, seg_idx, segment)
                    last_emit = time.perf_counter()

                for ch in llm.stream(history, cancel_token=token):
                    if ch.text_delta:
                        print(ch.text_delta, end="", flush=True)
                        assistant_parts.append(ch.text_delta)
                        buffer += ch.text_delta

                        while True:
                            split_at = None
                            for idx, ch_ in enumerate(buffer):
                                if ch_ not in split_chars:
                                    continue
                                if idx + 1 < min_chars:
                                    continue
                                split_at = idx
                                break
                            if split_at is None:
                                break
                            emit_segment(buffer[: split_at + 1])
                            buffer = buffer[split_at + 1 :]

                        if len(buffer) >= max_chars:
                            emit_segment(buffer[:max_chars])
                            buffer = buffer[max_chars:]

                        if buffer and (time.perf_counter() - last_emit) >= max_wait_s:
                            emit_segment(buffer)
                            buffer = ""
                    if ch.is_final:
                        break
                llm_ms = (time.perf_counter() - llm_start) * 1000.0

                assistant_text = "".join(assistant_parts).strip()
                print("")
                if not assistant_text:
                    continue

                if buffer:
                    emit_segment(buffer)
                    buffer = ""

                history.append(
                    LLMMessage(role="assistant", parts=[MessagePart(type="text", text=assistant_text)])
                )
                history = trim_history(history)
                print(f"[llm#R{reply_id}] done in {llm_ms:.1f} ms")

            except CancelledError:
                sd.stop()
                print("\n[llm] (interrupted)")
            except Exception as e:
                print(f"\n[llm/tts] (error) {e}")

    def tts_worker() -> None:
        while not stop_event.is_set():
            try:
                seg_gen_id, reply_id, seg_idx, text = tts_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if seg_gen_id != get_gen_id():
                print(f"[tts#R{reply_id}] drop seg {seg_idx} (stale gen_id={seg_gen_id})")
                continue

            try:
                print(f"[tts#R{reply_id}] start seg {seg_idx}")
                tts_start = time.perf_counter()
                res = tts.synthesize(text)
                tts_ms = (time.perf_counter() - tts_start) * 1000.0
                if seg_gen_id != get_gen_id():
                    print(f"[tts#R{reply_id}] drop seg {seg_idx} (stale gen_id={seg_gen_id})")
                    continue
                audio_q.put(
                    (seg_gen_id, reply_id, seg_idx, res.audio_bytes, res.sample_rate, tts_ms, len(text))
                )
                print(f"[tts#R{reply_id}] done seg {seg_idx} ({tts_ms:.1f} ms)")
            except Exception as e:
                print(f"\n[tts] (error) {e}")

    def playback_worker() -> None:
        buffer = PlaybackBuffer()
        stream: sd.OutputStream | None = None
        current_sr: int | None = None
        channels = 1

        def callback(outdata, frames, _time_info, _status) -> None:
            data, consumed = buffer.pop(frames, channels)
            outdata[:] = data
            if consumed:
                buffer.consume(consumed)

        while not stop_event.is_set():
            if playback_reset.is_set():
                buffer.clear()
                playback_reset.clear()
            try:
                seg_gen_id, reply_id, seg_idx, audio_bytes, sample_rate, tts_ms, text_len = (
                    audio_q.get(timeout=0.2)
                )
            except queue.Empty:
                for seg in buffer.pop_completed():
                    print(
                        f"[play#R{seg.reply_id}] done seg {seg.seg_idx} "
                        f"(TTS {seg.tts_ms:.1f} ms, chars {seg.text_len})"
                    )
                continue

            if seg_gen_id != get_gen_id():
                print(f"[play#R{reply_id}] drop seg {seg_idx} (stale gen_id={seg_gen_id})")
                continue

            try:
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if data.ndim == 2:
                    data = data.mean(axis=1)
                data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

                if current_sr != sample_rate:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                        stream = None
                    current_sr = sample_rate

                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=current_sr,
                        channels=channels,
                        dtype="float32",
                        callback=callback,
                    )
                    stream.start()

                buffer.push(
                    data,
                    SegmentState(
                        reply_id=reply_id,
                        seg_idx=seg_idx,
                        frames_left=len(data),
                        tts_ms=tts_ms,
                        text_len=text_len,
                    ),
                )
                print(f"[play#R{reply_id}] start seg {seg_idx}")
            except Exception as e:
                print(f"\n[playback] (error) {e}")

        if stream is not None:
            stream.stop()
            stream.close()

    t_asr = threading.Thread(target=asr_listener_loop, daemon=True)
    t_llm = threading.Thread(target=llm_tts_loop, daemon=True)
    t_tts = threading.Thread(target=tts_worker, daemon=True)
    t_play = threading.Thread(target=playback_worker, daemon=True)
    t_asr.start()
    t_llm.start()
    t_tts.start()
    t_play.start()

    try:
        while True:
            t_asr.join(timeout=1.0)
            t_llm.join(timeout=1.0)
            t_tts.join(timeout=1.0)
            t_play.join(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()
        interrupt.cancel()
        print("\n👋 bye")


if __name__ == "__main__":
    main()
