# 训练与推理流水线

## 数据准备阶段
1. 调用 `scripts/video_dataset/prepare_*.py`，将原始 JSON 样本转成 `input_ids`、`labels` 等张量并保存为 `.pt` 文件。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L23-L111】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L23-L111】
2. 标签前缀被置为 `IGNORE_INDEX`，训练时只回传回答部分的梯度，避免模型“抄题”。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L102-L107】

## 配置加载
- `option.get_args_parser()` 在 `CLI.py` 启动时立即解析所有路径与超参数，保证训练、推理脚本使用一致配置。【F:CLI.py†L31-L32】
- 其中包含 LoRA、VQ-VAE、多模态塔等模块的关键超参，详见《配置系统说明》。【F:options/option.py†L8-L88】

## 模型与权重初始化
1. 通过 `EmptyInitOnDevice` 与 `lora(...)` 上下文构建 Vicuna 基座，并启用 LoRA 低秩适配结构。【F:CLI.py†L222-L245】
2. 加载三类权重：Vicuna 预训练模型、LoRA 增量权重以及视觉投影器（MLP），随后将其合并。【F:CLI.py†L260-L273】
3. `get_processor` 根据配置创建 LanguageBind 感知塔和对应的预处理器，用于在 GPU 上提取视频特征。【F:CLI.py†L147-L170】

## 推理 Pipeline
1. 用户提供视频路径后，`video_processor` 完成采样与归一化，输出 `(B, C, T, H, W)` 的张量。【F:CLI.py†L289-L297】
2. `LlavaMetaModel` 将视频张量编码为 `(B, N, 1024)`，再经 `mm_projector` 投影到语言模型维度 `(B, N, 4096)`。【F:CLI.py†L135-L145】
3. 根据用户问题构造文本 prompt，并与视觉特征一起组织成 `generate` 所需的元组输入。【F:CLI.py†L305-L326】
4. `generate` 函数逐 token 采样，支持温度与 top-k 调节；当采样到 `<eos>` 或达到最大步数时终止。【F:generate.py†L14-L108】
5. 最终输出经过 tokenizer 解码并展示给用户。【F:CLI.py†L327-L331】

## 训练思路
虽然仓库中未包含完整的训练脚本，但利用生成的 `.pt` 数据与 LoRA 配置，可复用 `lit_gpt` 的训练循环：
1. 构建 `DataLoader` 读取 `.pt` 样本，使用 `input_ids` 与 `labels` 进行监督微调。
2. 只对 LoRA 参数和多模态投影器求梯度，保证基础模型稳定。【F:CLI.py†L222-L273】
3. 在评估阶段复用上述推理流水线，检验模型对视频问题的回答质量。
