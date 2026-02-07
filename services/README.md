# ai_core FastAPI Services (HTTPS)

## Services
- `asr`: `/v1/asr/transcribe` (input WAV, output JSON)
- `tts`: `/v1/tts/synthesize` (input JSON text, output `audio/wav`)
- `llm`: `/v1/llm/generate` and `/v1/llm/stream`
- `recorder`: `/v1/recorder/capture` (microphone capture, output `audio/wav`)

## Install
```bash
pip install -r services/requirements.txt
```

## Dev certificate (self-signed)
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
  -keyout certs/dev.key -out certs/dev.crt \
  -subj "/CN=localhost"
```

## Run HTTPS
```bash
python3 -m services.run_service asr --port 8443 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service tts --port 8444 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service llm --port 8445 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service recorder --port 8446 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
```

## TTS backends
- local simple model: `genie_tts` (runs in `ai_core` environment)
- isolated complex model: `gpt_sovits_remote` (runs in dedicated conda env + HTTPS service)

See `src/tts/GPT_Sovits_tts/REMOTE_SETUP.md` for GPT-SoVITS isolation details.

## Quick calls
ASR (`audio/wav` upload):
```bash
curl -k -X POST "https://127.0.0.1:8443/v1/asr/transcribe" \
  -F "audio=@out/test.wav;type=audio/wav" \
  -F "backend=paraformer"
```

TTS (`audio/wav` output, local Genie):
```bash
curl -k -X POST "https://127.0.0.1:8444/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"backend":"genie_tts","text":"你好，这是测试。"}' \
  --output out/tts_from_service.wav
```

LLM generate:
```bash
curl -k -X POST "https://127.0.0.1:8445/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{"backend":"qwen_official","messages":[{"role":"user","content":"你好"}]}'
```

Recorder capture (`audio/wav` output):
```bash
curl -k -X POST "https://127.0.0.1:8446/v1/recorder/capture" \
  -H "Content-Type: application/json" \
  -d '{"sample_rate":16000,"enable_segmenter":true}' \
  --output out/recorded.wav
```
