#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPT_DIR="$ROOT_DIR/src/tts/GPT_Sovits_tts/GPT-SoVITS"
ENV_NAME="${GPT_SOVITS_ENV_NAME:-GPTSoVits}"
HOST="${GPT_SOVITS_HOST:-0.0.0.0}"
PORT="${GPT_SOVITS_PORT:-9450}"
CFG_PATH="${GPT_SOVITS_TTS_CONFIG:-GPT_SoVITS/configs/tts_infer.yaml}"
CERT="${GPT_SOVITS_SSL_CERTFILE:-$ROOT_DIR/certs/dev.crt}"
KEY="${GPT_SOVITS_SSL_KEYFILE:-$ROOT_DIR/certs/dev.key}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

pushd "$GPT_DIR" >/dev/null
conda run -n "$ENV_NAME" python api_v2.py \
  -a "$HOST" \
  -p "$PORT" \
  -c "$CFG_PATH" \
  --ssl-certfile "$CERT" \
  --ssl-keyfile "$KEY"
popd >/dev/null
