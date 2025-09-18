# 配置系统说明

MotionLLM 的命令行参数集中在 `options/option.py`，使用 `argparse` 定义训练、推理与多模态组件的超参数集合。【F:options/option.py†L3-L88】

## 数据与路径参数
- `--prompt`：默认任务描述，用于构造指令模板时的初始文案。【F:options/option.py†L8-L16】
- `--vqvae_pth`、`--lora_path`、`--mlp_path` 等路径参数用于加载预训练 VQ-VAE、LoRA 适配器与多模态投影器权重。【F:options/option.py†L15-L20】
- `--data_dir` 设定预处理数据所在目录，以便 `prepare` 与训练脚本统一读取。【F:options/option.py†L15-L20】

## LoRA 与语言模型设置
- `--lora_r`、`--lora_alpha`、`--lora_dropout` 控制低秩适配层的秩、缩放及 dropout 概率，文件中增加的注释强调了它们和语言模型微调的关系。【F:options/option.py†L23-L28】
- `--block_size` 指定语言模型的上下文窗口长度，以保证与数据处理中 `max_seq_length` 的兼容性。【F:options/option.py†L30-L31】

## 训练超参数
- `--batch_size`、`--micro_batch_size`、`--weight_decay`、`--warmup_steps` 等参数用于控制优化器与调度器的行为。【F:options/option.py†L33-L44】
- `--learning_rate_lora` 与 `--learning_rate_mlp` 分别作用于语言模型 LoRA 层与投影 MLP，允许差异化学习率策略。【F:options/option.py†L33-L39】

## VQ-VAE 与量化设置
- 代码包含 VQ-VAE 的结构超参，如 `--code_dim`、`--down_t`、`--depth` 等，支持在不同分辨率与码本配置间切换。【F:options/option.py†L46-L58】
- `--quantizer` 及 `--quantbeta` 控制码本更新策略，适配不同的量化变体。【F:options/option.py†L60-L62】

## 可视化与扩展标志
- `--render` 与 `--motion_vq_token_path` 用于驱动 SMPL 渲染或载入离线离散化后的动作 token。【F:options/option.py†L64-L66】
- `--projectionnn`、`--diverse`、`--vinilla` 等标志打开特定的推理模式，如使用多层感知机投影或多样化描述。【F:options/option.py†L72-L74】

## 多模态塔配置
- `--image_tower` 与 `--video_tower` 决定使用的感知骨干，默认接入 LanguageBind 的图像/视频编码器。【F:options/option.py†L77-L83】
- `--mm_projector_type`、`--mm_hidden_size`、`--hidden_size` 控制视觉特征如何映射到语言模型的隐藏维度，与多模态投影器构造直接相关。【F:options/option.py†L77-L83】

以上参数在运行 `CLI.py` 或数据处理脚本时通过 `option.get_args_parser()` 统一加载，保证配置的一致性。【F:CLI.py†L31-L32】
