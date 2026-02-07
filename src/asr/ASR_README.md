ASR 模块扩展说明（给未来的自己）

====================
一、整体设计思路
====================

本项目中的 ASR（语音识别）模块采用“接口（base） + 工厂（factory）”的结构，
目的是让主循环（always_listen 等）与具体 ASR 模型完全解耦。

主程序只关心一件事：

    result = asr.transcribe(audio)
    text = result.text

不关心：
- 使用的是 Whisper / Paraformer / 其他模型
- 是否使用 GPU
- 模型内部参数与实现细节


====================
二、ASR 接口（base）
====================

文件：
    src/asr/base.py

核心思想：
    任何 ASR 模型，都必须提供一个方法：

        transcribe(audio, sample_rate=16000) -> ASRResult

其中：
- audio 可以是 wav 路径 或 numpy 音频
- ASRResult 是一个“统一的结果容器”

当前 ASRResult 定义为：

    ASRResult(
        text: str,          # 识别出的文本（核心字段）
        lang: Optional[str] # 语言（可选）
        backend: str        # 使用的 ASR 后端标识
    )

注意：
- base 不负责推理
- base 只是“接口规范 / 约定”
- 不要求继承某个父类，只要方法签名和返回类型一致即可


====================
三、ASR 工厂（factory）
====================

文件：
    src/asr/factory.py

作用：
- 统一管理“当前使用哪个 ASR”
- 让主程序不需要关心具体类路径

主程序中只允许通过 factory 创建 ASR：

    asr = create_asr("whisper")
    # 或
    asr = create_asr("paraformer")

也可以传入配置对象：

    from src.asr.whisper.config import ASRConfig
    asr = create_asr("whisper", ASRConfig(model_size="small"))

禁止在主循环（always_listen 等）中：
- 直接 new WhisperASR / ParaformerASR
- 写 if/else 判断模型类型


====================
四、如何新增一个 ASR 模型（步骤）
====================

假设要新增一个模型：MyNewASR

步骤 1：新建文件
-----------------
在 `src/asr/` 下新建一个模型文件夹：

    src/asr/mynew/config.py
    src/asr/mynew/model.py


步骤 2：实现 ASR 类
-------------------
在类中实现以下方法：

    def transcribe(self, audio, sample_rate=16000) -> ASRResult

要求：
- 必须返回 ASRResult
- 至少填写 text 字段
- backend 字段建议填写模型名，便于调试


步骤 3：接入 factory
--------------------
在 `src/asr/factory.py` 中：

1) import 新的 ASR 类
2) 在 create_asr(...) 中新增一个分支

例如：

    if name == "mynew":
        return MyNewASR(...)


步骤 4：切换使用
----------------
在 `always_listen.py` 或主入口中：

    asr = create_asr("mynew")

其余代码不需要修改。


====================
五、设计原则（重要）
====================

1) 主循环只依赖 ASR 接口（transcribe + ASRResult），不依赖具体模型
2) ASR 模块只负责“语音 -> 文本”，不掺杂 LLM / TTS / 业务逻辑
3) 新增或替换 ASR 时，不允许修改主循环逻辑
4) 所有模型差异，必须被封装在各自的 ASR 实现内部

遵守以上规则，可以保证 ASR 模块长期可扩展、可维护。
