# GPT-SoVITS Isolation (Complex Model)

`gpt_sovits_remote` is treated as a complex model and runs in a dedicated conda environment.
`genie_tts` remains local in the `ai_core` environment.

## 2) Prepare HTTPS cert
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
  -keyout certs/dev.key -out certs/dev.crt \
  -subj "/CN=localhost"
```

## 3) Start GPT-SoVITS HTTPS service
```bash
bash src/tts/GPT_Sovits_tts/run_https.sh
```
Default endpoint: `https://127.0.0.1:9450/tts`

## 4) Use from ai_core
```bash
export GPT_SOVITS_ENDPOINT="https://127.0.0.1:9450/tts"
export GPT_SOVITS_VERIFY_SSL=0
python3 - <<'PY'
from src.tts.factory import create_tts

tts = create_tts("gpt_sovits_remote")
res = tts.synthesize("你好，这是 gpt-sovits 远程调用测试。")
open("out/gpt_sovits_remote.wav", "wb").write(res.audio_bytes)
print(res.sample_rate, res.backend)
PY
```

## Optional env vars
- `GPT_SOVITS_ENV_NAME` (default: `GPTSoVits`)
- `GPT_SOVITS_HOST` (default: `0.0.0.0`)
- `GPT_SOVITS_PORT` (default: `9450`)
- `GPT_SOVITS_SSL_CERTFILE` / `GPT_SOVITS_SSL_KEYFILE`
- `GPT_SOVITS_TTS_CONFIG` (default: `GPT_SoVITS/configs/tts_infer.yaml`)
- `GPT_SOVITS_REF_AUDIO_PATH`, `GPT_SOVITS_PROMPT_TEXT`, `GPT_SOVITS_TEXT_LANG`, `GPT_SOVITS_PROMPT_LANG`
