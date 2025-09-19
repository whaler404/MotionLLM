# 数据处理

## 指令数据转换流程
1. 从 `/comp_robot/.../video_llava_{split}.json` 等路径加载包含 `instruction`、可选 `input` 以及 `output` 字段的原始 JSON（`scripts/video_dataset/prepare_video_dataset_video_llava.py:23-60`）。
2. 使用 `generate_prompt_mlp` 生成对话式提示模版；当存在视频路径时会插入 `INPUT_VIDEO` 字段（`scripts/video_dataset/prepare_video_dataset_video_llava.py:124-138`）。
3. 采用 LLaMA 分词器对提示及“提示+回答”进行编码，截断至 `max_seq_length`，并在完整响应后追加 EOS（`scripts/video_dataset/prepare_video_dataset_video_llava.py:85-111`）。
4. 将提示部分的标签位设置为 `IGNORE_INDEX=-1`，确保监督损失仅覆盖助手响应（`scripts/video_dataset/prepare_video_dataset_video_llava.py:103-108`）。
5. 将处理后的样本保存为 `.pt` 文件，方便后续微调快速加载；InternVideo 数据脚本遵循相同格式（`scripts/video_dataset/prepare_video_dataset_video_llava.py:58-60`、`scripts/video_dataset/prepare_video_dataset_intern_video.py:58-60`）。

## 提示模板
- 训练与推理阶段共用同一对话模板，保证标注与运行时的输入保持一致（`scripts/video_dataset/prepare_video_dataset_video_llava.py:124-160`、`app.py:324-328`、`CLI.py:308-312`）。
- 当需要系统提示时，可通过 `generate_system_command` 注入统一的系统消息（`scripts/video_dataset/prepare_video_dataset_video_llava.py:169-172`）。

## 推理阶段的模态预处理
- `get_processor` 会根据配置加载 LanguageBind 图像/视频塔及其预处理器，并将权重迁移到 CUDA FP16 模式（`app.py:277-296`、`CLI.py:147-256`）。
- 视频输入经采样、缩放与归一化后，转换为 `(B, C, T, H, W)` 的张量，再送入对应的塔编码（`app.py:310-318`、`CLI.py:289-304`）。
- 编码得到的特征网格会通过多模态投影器映射到 Vicuna 隐层宽度，以便与文本 token 拼接后送入解码器（`app.py:317-335`、`CLI.py:299-326`）。

## 动作（Motion）token 管线
- 可选的动作分支使用 VQ-VAE 对动作序列进行离散化。输入先维度置换为 `(batch, channels, timesteps)`，经编码、量化后再解码重建（`models/vqvae.py:37-92`）。
- 不同数据集（如 `t2m`、`kit`）会调整编码器/解码器的输入维度与关节数量（`models/vqvae.py:25-34`、`models/vqvae.py:112-124`）。
- 通过 `args.quantizer` 选择 EMA、重置等量化策略，`Quantize*` 模块提供不同的更新方式（`models/vqvae.py:24-34`、`options/option.py:60-62`）。

## 持久化内容
- 每个样本保存 `input_ids`、`input_ids_no_response`、`labels` 与可选 `sys_command`，以便训练期间恢复完整上下文与监督目标（`scripts/video_dataset/prepare_video_dataset_video_llava.py:111-112`）。
