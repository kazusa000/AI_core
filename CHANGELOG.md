# 更新日志

## 2026-02-07

### 1) 服务端体系建立
- 建立并统一了 ASR / TTS / LLM / Recorder 四类 FastAPI 服务。
- 外部项目（`ai_live`）可通过 HTTPS 统一调用 `ai_core` 能力。
- 服务接口职责明确，调用链路清晰，支持后续独立扩展。

### 2) 环境治理落地
- 主环境采用“可重建”策略，新增依赖清单与重建脚本（`requirements/core.in`、`requirements/core.txt`、`scripts/env/rebuild_core.sh`）。
- 复杂模型与主环境解耦，降低依赖污染和长期维护成本。
- 模型下线流程文档化，支持稳定清理与回滚。

### 3) `runtime_type` 机制建立
- 在模型注册层引入 `runtime_type`（`local` / `remote_managed`）。
- 当前仅 `gpt_sovits_remote` 为 `remote_managed`，其余模型保持 `local`。
- 服务侧已接入统一运行时判断，可按模型类型执行本地调用或远端进程拉起逻辑。
