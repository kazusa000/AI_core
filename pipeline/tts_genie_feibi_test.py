from __future__ import annotations

def main() -> None:
    from src.tts.factory import create_tts
    from src.tts.Genie_tts import GenieTTSConfig

    tts = create_tts("genie_tts", GenieTTSConfig())

    output_path = "out/feibi_tts.wav"
    text = "你好，我是菲比。今天的风有点大。"

    res = tts.synthesize(text)

    with open(output_path, "wb") as f:
        f.write(res.audio_bytes)

    print(f"Saved: {output_path} ({res.sample_rate} Hz)")


if __name__ == "__main__":
    main()
