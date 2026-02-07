# pipeline/asr_test.py
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from src.recorder import Recorder, RecorderConfig
from src.asr.factory import create_asr
from src.asr.base import ASRResult


@dataclass
class ListenLoopConfig:
    keep_latest_only: bool = True   # True=ASR忙时只保留最新一句（推荐）
    max_queue: int = 3              # keep_latest_only=False 时队列长度
    min_sec: float = 0.25
    enable_segmenter: bool = True


def main():
    cfg = ListenLoopConfig()

    # 只改这一行就能切模型
    asr = create_asr("paraformer")  # "whisper" or "paraformer"

    recorder = Recorder(
        RecorderConfig(
            sample_rate=16000,
            frame_ms=20,
            enable_segmenter=cfg.enable_segmenter,
            chunk_sec=4.0,
        )
    )

    # 队列：监听线程 -> ASR 线程
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=cfg.max_queue)

    print("🎧 Always listening...（Ctrl+C 退出）")
    print("说话 → 停顿 → 自动识别一句话（ASR 忙也会继续听）\n")

    stop_event = threading.Event()

    def push_audio(audio: np.ndarray) -> None:
        """根据策略入队：要么排队，要么只保留最新。"""
        if not cfg.keep_latest_only:
            # 正常排队：满了就丢最新（也可改成阻塞）
            try:
                audio_q.put_nowait(audio)
            except queue.Full:
                pass
            return

        # keep_latest_only=True：队列满了就清空旧的，只留最新
        while True:
            try:
                audio_q.put_nowait(audio)
                return
            except queue.Full:
                try:
                    audio_q.get_nowait()  # 丢掉最旧的一句
                except queue.Empty:
                    return

    def listener_loop():
        """永远监听，不被 ASR 阻塞。"""
        while not stop_event.is_set():
            audio = recorder.listen()
            duration = len(audio) / recorder.sample_rate
            if duration < cfg.min_sec:
                continue
            push_audio(audio)

    def asr_loop():
        """慢的部分：ASR 推理。"""
        while not stop_event.is_set():
            try:
                audio = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                res: ASRResult = asr.transcribe(audio, sample_rate=recorder.sample_rate)
                text = (res.text or "").strip()
                if not text:
                    continue

                lang = res.lang or "-"
                backend = res.backend or asr.__class__.__name__
                print(f"\n[{backend}] [lang={lang}] {text}")

            except Exception as e:
                print(f"\n[ASR ERROR] {e}")

    t_listen = threading.Thread(target=listener_loop, daemon=True)
    t_asr = threading.Thread(target=asr_loop, daemon=True)
    t_listen.start()
    t_asr.start()

    try:
        while True:
            # 主线程保持活着，Ctrl+C 在这里捕获
            t_listen.join(timeout=1.0)
            t_asr.join(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()
        print("\n👋 bye")


if __name__ == "__main__":
    main()
