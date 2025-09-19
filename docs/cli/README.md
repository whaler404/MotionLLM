# MotionLLM 命令行使用文档

## 概述

本目录包含 MotionLLM 命令行界面（CLI）的详细使用文档。CLI 是 MotionLLM 的主要交互方式，支持多模态（视频、图像、文本）的推理和训练任务。

## 快速开始

### 基本使用
```bash
# 运行基本的视频问答系统
python CLI.py \
    --lora_path ./finetuned_models/video_qa_lora.pth \
    --mlp_path ./finetuned_models/video_mlp_projector.pth

litgpt download lmsys/vicuna-7b-v1.5

litgpt convert_to_litgpt checkpoints/vicuna-7b-v1.5

python CLI.py --lora_path ./checkpoints/MotionLLM-7B/motionllm-ckpt/lora.pth --mlp_path ./checkpoints/MotionLLM-7B/motionllm-ckpt/linear.pth --device_map "llm=cuda:0,image=cuda:1,video=cuda:2,projector=cuda:3"

python CLI_hf.py --mlp_path ./checkpoints/MotionLLM-7B/motionllm-ckpt/linear.pth --device_map "llm=cuda:0,image=cuda:1,video=cuda:2,projector=cuda:3"
```

### 使用 Hugging Face 版本 CLI
`CLI_hf.py` 基于 Hugging Face Transformers 加载 Vicuna-7B 及 LoRA 适配器，可直接复用同一套参数：

```bash
python CLI_hf.py \
    --lora_path ./hf_adapters/video_qa_adapter \  # 需为 Hugging Face/PEFT 目录
    --mlp_path ./finetuned_models/video_mlp_projector.pth \
    --device_map "llm=0,video=1,projector=2"
```

- Vicuna-7B 基座将从 `checkpoints/vicuna-7b-v1.5/` 的 Hugging Face 权重加载。
- `--lora_path` 需指向包含 `adapter_config.json` 的 PEFT LoRA 目录；若仍为 `.pth` 权重，请先转换为 Hugging Face 兼容格式。
- 其他多模态塔与投影器的加载流程与原 CLI 一致，仍可使用 `--device_map` 精细控制显卡分配。

### 交互式界面
启动后，CLI 会提供交互式界面：
```bashli
Input video path: ./examples/sample_video.mp4
Your question: 视频中的人在做什么？
================================
Model output: 视频中一个人正在走路...
================================
```

## 文档结构

### 核心文档

- **[使用概览](./overview.md)** - CLI 功能特性、系统架构和使用场景介绍
- **[模型配置](./model_configuration.md)** - 详细的模型路径、检查点和参数配置说明
- **[使用示例和最佳实践](./usage_examples.md)** - 实际应用场景示例和优化技巧

### 快速导航

#### 如果您需要：
- **了解CLI的基本功能** → 阅读 [使用概览](./overview.md)
- **配置模型和参数** → 阅读 [模型配置](./model_configuration.md)
- **查看实际使用示例** → 阅读 [使用示例和最佳实践](./usage_examples.md)
- **学习性能优化** → 阅读 [使用示例和最佳实践](./usage_examples.md#性能优化技巧)
- **部署到生产环境** → 阅读 [使用示例和最佳实践](./usage_examples.md#部署和集成)

## 主要特性

### 支持的功能
- **视频问答**: 输入视频和问题，生成文本答案
- **图像描述**: 输入图像，生成描述性文本
- **多模态融合**: 结合多种模态进行理解
- **交互式对话**: 支持多轮对话

### 技术特点
- **多模态编码**: 支持 LanguageBind 编码器
- **LoRA微调**: 支持轻量级模型适配
- **灵活配置**: 丰富的参数配置选项
- **高性能**: 支持GPU加速和内存优化

## 必需文件

### 模型文件结构
```
checkpoints/
└── vicuna-7b-v1.5/
    ├── lit_model.pth          # 预训练LLM权重 (13GB)
    └── tokenizer.model        # 分词器模型 (4MB)

finetuned_models/
├── video_qa_lora.pth         # LoRA微调权重 (100MB-1GB)
└── video_mlp_projector.pth  # 多模态投影器权重 (10-100MB)
```

### 配置文件
```
options/
├── option.py                 # 主要参数配置
├── option_video.py           # 视频相关配置
└── option_video_model.py     # 视频模型配置
```

## 关键参数

### 模型路径参数
```bash
--lora_path              # LoRA微调模型路径 (必需)
--mlp_path               # 多模态投影器路径 (必需)
--vqvae_pth              # VQ-VAE模型路径 (可选)
--resume_pth             # 恢复训练检查点 (可选)
```

### 多模态配置
```bash
--video_tower            # 视频编码器: LanguageBind/LanguageBind_Video_merge
--image_tower            # 图像编码器: LanguageBind/LanguageBind_Image
--mm_projector_type      # 投影器类型: mlp2x_gelu/linear/qformer2_64
--mm_hidden_size         # 多模态特征维度: 1024
--hidden_size            # 语言模型维度: 4096
```

### LoRA参数
```bash
--lora_r                 # LoRA秩: 64
--lora_alpha             # LoRA缩放因子: 16
--lora_dropout           # LoRA dropout: 0.05
```

### 推理参数
```bash
--max_new_tokens         # 最大生成token数: 200
--temperature            # 采样温度: 0.8
--top_k                  # top-k采样: 200
--dtype                  # 数据类型: float32
```

## 使用场景

### 1. 基础视频问答
```bash
python CLI.py \
    --lora_path ./models/video_qa_lora.pth \
    --mlp_path ./models/video_mlp_projector.pth
```

### 2. 体育动作分析
```bash
python CLI.py \
    --lora_path ./models/sports_analysis_lora.pth \
    --mlp_path ./models/sports_mlp_projector.pth \
    --mm_projector_type qformer2_64 \
    --max_new_tokens 300 \
    --temperature 0.6
```

### 3. 医疗康复评估
```bash
python CLI.py \
    --lora_path ./models/medical_rehab_lora.pth \
    --mlp_path ./models/medical_mlp_projector.pth \
    --mm_projector_type qformer2_64 \
    --max_new_tokens 250 \
    --temperature 0.4
```

## 性能优化

### 内存优化
```bash
# 低内存配置
--lora_r 32                    # 减少LoRA秩
--micro_batch_size 1           # 减少批次大小
--mm_projector_type linear      # 使用线性投影器
```

### 性能优化
```bash
# 高性能配置
--lora_r 128                   # 增加LoRA秩
--mm_projector_type mlp3x_gelu  # 使用3层MLP
--max_new_tokens 300           # 增加输出长度
```

## 故障排除

### 常见问题

#### 内存不足
```bash
# 解决方案
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# 使用较小的LoRA秩
--lora_r 32
```

#### 模型加载失败
```bash
# 检查文件
ls -la ./checkpoints/vicuna-7b-v1.5/
ls -la ./finetuned_models/
```

#### 视频处理错误
```bash
# 检查视频格式
ffmpeg -i your_video.mp4
# 确保视频编码正确
```

### 调试工具

#### 模型维度检查
```python
# 在代码中添加调试信息
print(f"投影器类型: {type(model.mm_projector).__name__}")
print(f"编码器隐藏大小: {model.mm_hidden_size}")
```

#### 内存使用分析
```python
# 监控GPU内存
import torch
print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

## 部署选项

### 本地部署
```bash
# 直接运行
python CLI.py --lora_path ./models/your_model.pth
```

### Web服务部署
参考 [使用示例](./usage_examples.md) 中的 Web 服务部署示例

### Docker部署
```bash
# 构建镜像
docker build -t motionllm-cli .

# 运行容器
docker run -p 5000:5000 motionllm-cli
```

## 高级用法

### 自定义配置文件
```python
# 创建自定义配置
config = {
    'lora_path': './models/custom_lora.pth',
    'mlp_path': './models/custom_mlp.pth',
    'mm_projector_type': 'qformer2_64',
    'max_new_tokens': 400
}
```

### 批量处理
```python
# 批量处理多个视频
video_list = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in video_list:
    result = process_video(video)
```

### API集成
```python
# 集成到现有应用
from motionllm import MotionLLMCLI

cli = MotionLLMCLI(config)
result = cli.analyze_video(video_path, question)
```

## 相关资源

- [MotionLLM 主文档](../README.md)
- [模型架构概览](../model_architecture.md)
- [多模态编码器文档](../models/multimodal_encoder/README.md)
- [多模态投影器文档](../models/multimodal_projector/README.md)

## 更新日志

- **2024-01-18**: 完成CLI使用文档
- **2024-01-17**: 创建CLI文档结构
- **2024-01-16**: 开始CLI功能分析

## 贡献

如果您发现文档中的错误或有改进建议，请提交 Issue 或 Pull Request。

## 许可证

本文档遵循 MotionLLM 项目的许可证条款。

---

**提示**: 首次使用时，建议先阅读 [使用概览](./overview.md) 了解基本功能，然后参考 [使用示例](./usage_examples.md) 进行实际操作。
