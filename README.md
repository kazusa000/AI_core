# ai_core

`ai_core` 是一个面向语音交互的核心服务工程，提供 ASR、LLM、TTS、Recorder 四类能力，并通过 HTTPS 对外提供统一接口。

## 核心结构
- `src/asr`：语音识别后端与工厂
- `src/llm`：大模型后端与工厂
- `src/tts`：语音合成后端与工厂
- `src/recorder`：录音与 VAD 切分能力
- `services`：FastAPI 服务入口（ASR/TTS/LLM/Recorder）
- `pipeline`：联调与端到端测试脚本
- `requirements`：主环境依赖与治理文档

## 快速启动（本地直调，不走 HTTP）
- `python3 -m pipeline.asr_test`：仅测试 ASR 识别流程（麦克风输入 -> 文本输出）。
- `python3 -m pipeline.asr_llm_stream`：测试 ASR + LLM 流式回复（不含 TTS 播放）。
- `python3 -m pipeline.asr_llm_tts_stream`：测试完整语音链路（ASR -> LLM -> TTS）。
- `python3 -m pipeline.tts_genie_feibi_test`：仅测试 Genie TTS 生成音频样本。

## 快速启动（HTTPS）
```bash
python3 -m services.run_service asr --port 8443 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service tts --port 8444 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service llm --port 8445 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
python3 -m services.run_service recorder --port 8446 --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key
```

## 运行模式
- `local`：模型在 `ai_core` 进程内调用
- `remote_managed`：模型由运行时管理器按需拉起独立进程后调用（当前用于 `gpt_sovits_remote`）

## 相关文档
- 服务说明：`services/README.md`
- 依赖治理：`requirements/README.md`
- 更新日志：`CHANGELOG.md`
