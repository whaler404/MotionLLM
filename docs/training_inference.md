# 训练与推理

## LoRA + 投影器训练流程
1. **数据准备**：使用提供的脚本将指令/回答 JSON 转换为张量，统一采用带有 `INPUT_VIDEO` 的对话模版，并用 `IGNORE_INDEX` 屏蔽提示部分（`scripts/video_dataset/prepare_video_dataset_video_llava.py:23-176`）。
2. **主干初始化**：从 Vicuna 7B 权重启动模型，并借助 `lit_llama.lora` 仅解冻需要训练的 Query/Value 投影（`app.py:508-528`、`CLI.py:224-243`）。
3. **多模态投影训练**：加载并冻结 LanguageBind 视觉塔，仅训练 `mm_projector` 与 LoRA 适配器，使视觉特征能够对齐 Vicuna 隐层（`app.py:537-555`、`models/multimodal_projector/builder.py:235-247`）。
4. **优化超参**：通过 `--learning_rate_lora` 与 `--learning_rate_mlp` 为 LoRA 和 MLP 设置独立学习率；其他批次、Warmup、日志频率等由 `options/option.py` 中的参数控制（`options/option.py:33-44`）。
5. **可选动作分支**：在需要动作文本联合建模时，实例化 VQ-VAE 及相关编码器，量化动作 Token 并联合训练描述头（`models/vqvae.py:7-134`、`models/modules.py:1-74`）。

## 评估工具
- `models/evaluator_wrapper.py` 会加载预训练的文本/动作编码器，用于计算嵌入相似度，支持检索式指标或动作文本对齐评估（`models/evaluator_wrapper.py:1-120`）。
- 通过 `--motionx_zero_shot_path` 等参数可以指定零样本评测集，复现论文中的 MotionX 评估（`options/option.py:69-70`）。

## 推理流程

### CLI 模式
1. 加载基础 Vicuna 权重、合并 LoRA，并恢复投影器检查点（`CLI.py:204-274`）。
2. 实例化 LanguageBind 视频处理器，将输入视频转为归一化后的张量（`CLI.py:255-304`）。
3. 编码视频帧并通过投影器映射到 Vicuna 宽度，再与分词后的提示前缀/后缀拼接（`CLI.py:299-326`）。
4. 调用自回归生成函数，输入包含文本 token 与视频特征的元组，最终解析出助手回答（`CLI.py:317-331`、`generate.py:40-108`）。

### Gradio 演示
- 与 CLI 设置一致，但采用 Gradio Blocks 进行人机交互，并在上传、缓存及反馈记录（点赞/差评）等环节做了封装（`app.py:277-620`）。
- 支持通过环境变量设置临时目录以及 Hugging Face 镜像，方便在受限网络环境中运行（`README.md:110-124`）。

## 运维提示
- 代码假定存在可用的 CUDA 设备，Lightning Fabric 会将模型转换为 BF16/FP16 以提升效率（`app.py:496-559`、`generate.py:146-160`）。
- 在推理时必须保证提示中提供的文件路径有效，否则 LanguageBind 处理器在读取帧时会失败（`CLI.py:289-296`、`app.py:310-318`）。

## 后续建议
- 若需要复现实验，可在仓库中加入完整的训练脚本或 Notebook，将上述流程自动化。
