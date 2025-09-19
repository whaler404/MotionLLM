# MotionLLM 命令行使用概览

## 概述

MotionLLM 提供了完整的命令行界面（CLI）用于模型推理和训练。CLI 脚本 `CLI.py` 实现了多模态（视频、图像、文本）的交互式推理功能，支持加载预训练模型、LoRA 微调模型以及多模态投影器。

## 功能特性

### 核心功能
- **多模态推理**: 支持视频、图像、文本的联合理解
- **交互式对话**: 命令行交互式问答界面
- **模型加载**: 支持预训练模型和LoRA微调模型
- **多模态编码**: 集成LanguageBind等多模态编码器
- **灵活配置**: 丰富的配置选项和参数设置

### 支持的模态
- **视频**: 通过LanguageBind Video编码器处理
- **图像**: 通过LanguageBind Image编码器处理
- **文本**: 基于LLaMA/Vicuna语言模型
- **运动**: 通过VQ-VAE处理运动数据

## 基本架构

### 系统组件
```
CLI.py
├── 模型加载系统
│   ├── 预训练LLM加载 (Vicuna-7B)
│   ├── LoRA权重加载
│   └── 多模态投影器加载
├── 多模态编码系统
│   ├── 视频编码器 (LanguageBind Video)
│   ├── 图像编码器 (LanguageBind Image)
│   └── 特征投影器
├── 预处理系统
│   ├── 视频帧采样
│   ├── 图像预处理
│   └── 文本分词
└── 推理系统
    ├── 特征编码
    ├── 多模态融合
    └── 文本生成
```

### 数据流
```
输入视频 → 预处理 → 视频编码器 → 投影器 → 多模态融合 → 语言模型 → 输出文本
用户问题 → 文本编码 →───────────────────────────────┘
```

## 主要参数配置

### 1. 模型路径参数
```bash
# 必需的模型路径参数
--lora_path              # LoRA微调模型路径
--mlp_path               # 多模态投影器路径

# 内置的模型路径（代码中硬编码）
pretrained_llm_path      # 基础LLM模型路径: ./checkpoints/vicuna-7b-v1.5/lit_model.pth
tokenizer_llm_path       # 分词器路径: ./checkpoints/vicuna-7b-v1.5/tokenizer.model
```

### 2. 多模态编码器参数
```bash
--image_tower            # 图像编码器: LanguageBind/LanguageBind_Image
--video_tower            # 视频编码器: LanguageBind/LanguageBind_Video_merge
--mm_vision_select_layer # 视觉编码器层选择: -2 (倒数第二层)
--mm_projector_type      # 投影器类型: mlp2x_gelu
--mm_hidden_size         # 多模态特征维度: 1024
--hidden_size            # 语言模型特征维度: 4096
```

### 3. LoRA微调参数
```bash
--lora_r                 # LoRA秩: 64
--lora_alpha             # LoRA缩放因子: 16
--lora_dropout           # LoRA dropout: 0.05
```

### 4. 推理参数
```bash
--max_new_tokens         # 最大生成token数: 200
--top_k                  # top-k采样: 200
--temperature            # 采样温度: 0.8
--dtype                  # 数据类型: float32
--accelerator            # 加速器: auto
```

## 模型文件结构

### 必需的检查点文件
```
checkpoints/
├── vicuna-7b-v1.5/
│   ├── lit_model.pth          # 预训练LLM权重
│   └── tokenizer.model        # 分词器模型
└── your_finetuned_models/
    ├── lora_adapter.pth       # LoRA微调权重
    └── mlp_projector.pth      # 多模态投影器权重
```

### 多模态编码器缓存
```
cache_dir/
├── LanguageBind_Image/       # 图像编码器缓存
├── LanguageBind_Video_merge/ # 视频编码器缓存
└── huggingface/             # HuggingFace缓存
```

## 使用场景

### 1. 视频问答
输入视频文件和问题，模型生成答案：
```bash
# 用户输入视频路径
Input video path: /path/to/your/video.mp4

# 用户输入问题
Your question: 视频中的人在做什么？
```

### 2. 图像描述
输入图像文件，生成描述性文本：
```bash
# 需要修改代码以支持图像模式
Input image path: /path/to/your/image.jpg

# 生成图像描述
Generated description: 一只猫坐在沙发上
```

### 3. 多模态对话
结合视频/图像和文本进行多轮对话：
```bash
# 第一轮
Input video path: /path/to/video.mp4
Your question: 视频中有什么动作？
Answer: 视频中一个人在走路

# 第二轮（同一视频）
Your question: 他穿着什么颜色的衣服？
Answer: 他穿着蓝色的衣服
```

## 性能要求

### 硬件要求
- **GPU**: 推荐 NVIDIA GPU with ≥ 8GB VRAM
- **内存**: 推荐 ≥ 16GB RAM
- **存储**: 需要约 15GB 存储空间用于模型文件

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+
- 相关依赖包（见requirements.txt）

## 输入输出格式

### 视频输入
- **格式**: MP4, AVI, MOV 等常见视频格式
- **分辨率**: 自动调整到 224×224
- **帧采样**: 自动采样 8 帧
- **预处理**: 标准化处理

### 文本输入
- **语言**: 中文或英文
- **编码**: UTF-8
- **长度**: 建议不超过 200 字符

### 文本输出
- **长度**: 由 `--max_new_tokens` 控制
- **语言**: 与输入语言一致
- **格式**: 自然语言文本

## 限制和注意事项

### 当前限制
1. **仅支持视频模式**: 当前代码主要针对视频问答设计
2. **固定模型架构**: 使用 Vicuna-7B 作为基础模型
3. **预定义路径**: 部分模型路径硬编码在代码中
4. **单GPU支持**: 目前仅支持单GPU推理

### 使用建议
1. **检查点准备**: 确保所有必需的模型文件都已下载
2. **内存管理**: 注意GPU内存使用，必要时调整批次大小
3. **视频格式**: 确保视频格式兼容，建议使用 MP4
4. **问题设计**: 设计清晰、具体的问题以获得更好的回答

## 故障排除

### 常见问题
1. **模型加载失败**: 检查模型路径是否正确
2. **GPU内存不足**: 减少批次大小或使用更小的模型
3. **视频处理失败**: 检查视频格式和编解码器
4. **生成质量差**: 调整采样参数（temperature, top_k）

### 调试技巧
1. **逐步验证**: 先测试基础模型加载，再测试多模态功能
2. **日志检查**: 观察运行时的警告和错误信息
3. **简化测试**: 使用简单的测试视频进行初步验证

## 扩展和定制

### 支持新模态
1. 添加对应的编码器配置
2. 修改预处理逻辑
3. 更新投影器配置

### 自定义模型
1. 修改模型加载逻辑
2. 调整特征维度配置
3. 更新投影器参数

### 性能优化
1. 实现模型量化
2. 添加批处理支持
3. 优化内存使用

MotionLLM CLI 提供了强大的多模态推理能力，通过合理的配置和使用，可以实现高质量的视频理解和问答功能。