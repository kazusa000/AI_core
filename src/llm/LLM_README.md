LLM 模块扩展说明（给未来的自己）

====================
一、整体设计思路
====================

LLM 模块采用“接口（base） + 工厂（factory） + 模型子目录”的结构，
目的是让主流程只依赖统一的消息结构和工厂入口。

主程序只关心：

    llm = create_llm("gemini")
    res = llm.stream(messages)

不关心：
- 具体后端 SDK
- 配置读取方式
- 模型内部实现细节


====================
二、LLM 接口（base）
====================

文件：
    src/llm/base.py

核心数据结构：
- LLMMessage: 由 role + parts 组成
- MessagePart: 支持 text / tool_call / tool_result
- ToolCall / ToolResult: 为工具调用预留结构化承载

说明：
- Gemini 适配会优先使用 SDK 的原生 part（function_call / function_response）。
- 如果 SDK 不支持，将回退为 JSON 文本传入模型。


====================
三、LLM 工厂（factory）
====================

文件：
    src/llm/factory.py

作用：
- 统一入口：create_llm(name, cfg)
- 用注册表管理后端（name -> ConfigClass + ModelClass）
- 保持工厂干净，不写具体配置细节


====================
四、模型目录结构
====================

每个后端一个子目录：

    src/llm/{backend}/
        config.py   # 默认配置和环境变量读取
        model.py    # 后端实现，适配 base 的接口
        __init__.py # 统一导出

示例（Gemini）：

    src/llm/Gemini/config.py
    src/llm/Gemini/model.py

示例（Qwen 官方 Transformers）：

    src/llm/Qwen_official/config.py
    src/llm/Qwen_official/model.py


====================
五、如何新增一个 LLM 模型（步骤）
====================

假设要新增一个模型：MyLLM

步骤 1：新建目录
-----------------

    src/llm/myllm/config.py
    src/llm/myllm/model.py

步骤 2：实现配置
-----------------
在 config.py 中定义 dataclass，提供默认值和环境变量读取。

步骤 3：实现模型
-----------------
在 model.py 中实现：

    def stream(messages, cancel_token=None) -> Iterator[LLMChunk]

要求：
- 读取 LLMMessage.parts，至少处理 text 部分
- 返回 LLMChunk（增量输出）
- 检查 cancel_token 并尽快中断

步骤 4：注册工厂
-----------------
在 src/llm/factory.py 的注册表中添加：

    LLM_REGISTRY["myllm"] = (MyConfig, MyLLM)


====================
六、设计原则（重要）
====================

1) 主流程只依赖 base + factory
2) 配置只放在各自模型目录
3) 新增模型不改业务逻辑
4) 工具调用和多模态能力用 MessagePart 扩展，不改上层调用方式
