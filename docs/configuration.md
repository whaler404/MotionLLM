# 配置指南

## 前置条件
- 将 Vicuna 1.5 7B 权重和分词器下载到 `./checkpoints/vicuna-7b-v1.5/`。Gradio 与 CLI 入口会先加载基础的 `lit_model.pth` 与分词器，然后再合并 LoRA 权重（参考 `app.py:488-563`、`CLI.py:204-283`）。
- 通过 `--lora_path` 与 `--mlp_path` 指定 LoRA 检查点以及多模态投影 MLP。两者都需要指向微调阶段生成的 `*.pth` 文件（`app.py:488-555`、`CLI.py:204-274`）。
- 可选：设置 `GRADIO_TEMP_DIR` 以指定临时目录，避免将上传内容存放在系统默认的 `/tmp`（`README.md:110-124`）。

## 命令行参数

### 数据与检查点
- `--dataname`、`--data_dir` 与 `--out_dir` 指定数据集根目录及输出路径（`options/option.py:11-20`）。
- `--vqvae_pth`、`--resume_pth`、`--motion_vq_token_path`、`--motionx_zero_shot_path` 在启用动作分支时提供 VQ-VAE 及零样本评测的相关权重与资源（`options/option.py:16-20`、`options/option.py:65-71`）。

### LoRA 与 LLM 适配
- `--lora_path`、`--lora_r`、`--lora_alpha`、`--lora_dropout` 控制低秩适配器位置与超参数。默认只训练注意力的 query/value 投影（`options/option.py:18-28`、`app.py:512-528`、`CLI.py:227-243`）。
- `--block_size` 设置文本上下文的最大长度（`options/option.py:30-31`）。

### 训练调度
- `--batch_size`、`--micro_batch_size`、`--learning_rate_lora`、`--learning_rate_mlp`、`--weight_decay`、`--warmup_steps`、`--eval_interval`、`--save_interval`、`--eval_iters`、`--log_interval` 决定优化器、学习率及日志频率（`options/option.py:33-44`）。

### VQ-VAE 动作编码
- `--code_dim`、`--nb_code`、`--mu`、`--down_t`、`--stride_t`、`--width`、`--depth`、`--dilation_growth_rate`、`--output_emb_width`、`--vq_act`、`--seed`、`--window_size` 控制 VQ-VAE 结构（`options/option.py:46-58`）。
- `--quantizer` 与 `--quantbeta` 用于切换量化器类型（`options/option.py:60-62`）。

### 多模态塔与投影层
- `--image_tower`、`--video_tower`、`--mm_vision_select_layer`、`--mm_projector_type`、`--mm_hidden_size`、`--hidden_size` 配置 LanguageBind 编码器及映射到 Vicuna 隐层宽度的投影模块（`options/option.py:77-83`、`models/multimodal_projector/builder.py:235-247`）。默认 `mlp2x_gelu` 为两层线性层加 GELU。

### 其他参数
- `--projectionnn`、`--diverse`、`--vinilla` 打开额外的生成策略或可视化分支（`options/option.py:72-74`）。
- `--model_type` 用于在保存或评测时标记模型类型（`options/option.py:85-86`）。

## 环境提示
- 无法直接访问 Hugging Face 时，可使用镜像：`HF_ENDPOINT=https://hf-mirror.com`（`README.md:118-121`）。
- 若在多卡环境运行，可通过 `CUDA_VISIBLE_DEVICES` 设定设备编号；代码假定存在可用的 CUDA 设备（`app.py:480`、`CLI.py:255-304`）。
