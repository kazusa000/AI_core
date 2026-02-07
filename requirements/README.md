# ai_core 依赖管理策略

`requirements/core.in` 定义 `ai_core/.venv` 的主依赖集合。  
`requirements/core.txt` 是重建环境时实际安装的入口文件。

## 一键重建主环境
```bash
bash scripts/env/rebuild_core.sh
```

## 为什么这样做
- 不手工逐个 `pip uninstall` 模型依赖。
- 依赖变更后直接重建 `.venv`，结果更干净、更可控。
- 复杂模型在独立环境中运行，不污染主环境。

## 删除模型的标准操作

### 1. 删除复杂模型（独立环境）
例：`gpt_sovits_remote`

1. 停掉模型服务
```bash
pkill -f "GPT-SoVITS/api_v2.py"
```
2. 删除模型代码/权重目录（确认不再使用后）
```bash
rm -rf src/tts/GPT_Sovits_tts/GPT-SoVITS
```
3. 删除独立 conda 环境 (所有复杂模型使用conda而不是venv)
```bash
conda env remove -n GPTSoVits
```
4. 从工厂注册中移除后端（如 `src/tts/factory.py`）

说明：复杂模型是隔离部署，删除时通常不需要重建 `ai_core/.venv`。

### 2. 删除简单模型（主环境内）

1. 从代码注册中移除后端（如 `src/tts/factory.py`）
2. 从 `requirements/core.in` 删除该模型相关依赖
3. 重建主环境
```bash
bash scripts/env/rebuild_core.sh
```

说明：简单模型依赖在主环境里，正确做法是“改依赖清单 + 重建环境”，而不是手动卸载。
