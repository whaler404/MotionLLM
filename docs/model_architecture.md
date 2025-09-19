# 模型架构

## 总览
- MotionLLM 以 LoRA 适配后的 Vicuna 7B 作为文本解码主干。加载阶段先恢复基础权重，再合并仅作用于 Query/Value 投影的 LoRA 参数（`app.py:508-556`、`CLI.py:224-274`）。
- 推理时使用 Lightning Fabric 管理设备与精度，在单张 GPU 上以 FP16/BF16 运行自回归生成（`app.py:496-559`、`generate.py:146-160`）。

## 视觉与视频塔
- `build_image_tower` 与 `build_video_tower` 根据配置选择 LanguageBind 或 CLIP 编码器；视频塔输出 `(frames × patches, hidden)` 的特征序列（`models/multimodal_encoder/builder.py:7-30`）。
- 编码器延迟加载自 Hugging Face，并暴露预处理器，保证输入张量的归一标准一致（`app.py:277-296`、`CLI.py:147-256`）。
- 所有塔默认冻结梯度，并将输出转换为模型所需的数据类型，以减少推理显存。 

## 多模态投影器
- `build_vision_projector` 根据 `mm_projector_type` 构建线性层、多层感知机、Identity 或基于 BLIP-2 的 Q-Former 等多种投影器（`models/multimodal_projector/builder.py:235-247`）。
- 默认的 `mlp2x_gelu` 通过两层线性层夹带 GELU 激活，将 LanguageBind 1024 维特征映射到 Vicuna 的 4096 维隐空间（`options/option.py:81-83`、`models/multimodal_projector/builder.py:235-247`）。

## 文本主干与生成机制
- Vicuna 主干来自 `lit_gpt.GPT`，自回归逻辑由 `generate.generate` 实现。生成函数支持“文本 token + 多模态特征”的组合输入，接收包含前缀、视频特征与后缀的元组（`generate.py:14-108`）。
- 为未来扩展，仓库预定义了多种模态的特殊标记（如 `<video>`、`<vi_patch>`）（`models/constants.py:7-17`）。

## 提示与对话桥接
- 提示模版遵循 Chat 格式，包含 `USER`/`ASSISTANT` 角色及 `INPUT_VIDEO` 占位符（`scripts/video_dataset/prepare_video_dataset_video_llava.py:124-138`）。
- 推理时先分别编码文本前缀与后缀，再插入投影后的视频特征，最终拼接成送入解码器的完整序列（`app.py:322-335`、`CLI.py:305-326`）。

## 动作分支组件
- VQ-VAE (`models/vqvae.py:7-134`) 对动作序列进行离散化，`HumanVQVAE` 根据数据集调整关节布局并提供编码/解码接口。
- `models/modules.py` 与 `models/evaluator_wrapper.py` 中的双向 GRU 编码器为评估或检索场景提供文本/动作嵌入能力（`models/evaluator_wrapper.py:1-120`）。
