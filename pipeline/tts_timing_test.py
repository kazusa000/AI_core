from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.tts.factory import create_tts
from src.tts.Genie_tts import GenieTTSConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TTS synthesis latency.")
    parser.add_argument(
        "--text",
        default="作为AI助手，你可以根据需要称呼我。",
        help="Text to synthesize.",
    )
    # 不过，在我们的对话中，你可以将我视为能够提供信息、帮助解答问题或执行任务的智能系统。如果你有任何具体问题或者需要帮助，请随时告诉我！。
    parser.add_argument(
        "--output",
        default="out/tts_timing.wav",
        help="Output WAV path.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    tts = create_tts("genie_tts", GenieTTSConfig())
    load_ms = (time.perf_counter() - load_start) * 1000.0

    infer_start = time.perf_counter()
    res = tts.synthesize(args.text)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(res.audio_bytes)

    print(f"Load time: {load_ms:.1f} ms")
    print(f"Infer time: {infer_ms:.1f} ms")
    print(f"Saved: {out_path} ({res.sample_rate} Hz)")


if __name__ == "__main__":
    main()
