# MotionLLM CLI命令行参数详细分析

基于 `python CLI.py --lora_path ./finetuned_models/video_qa_lora.pth --mlp_path ./finetuned_models/video_mlp_projector.pth` 命令的完整参数分析

## 命令行参数值详细列表

### 1. 用户明确指定的参数

| 参数 | 值 | 来源 |
|------|----|----|
| `--lora_path` | `./finetuned_models/video_qa_lora.pth` | 用户指定 |
| `--mlp_path` | `./finetuned_models/video_mlp_projector.pth` | 用户指定 |

### 2. 系统硬编码的路径参数

#### 基础LLM模型路径
```python
# CLI.py 第205-206行
pretrained_llm_path = "./checkpoints/vicuna-7b-v1.5/lit_model.pth"
tokenizer_llm_path = "./checkpoints/vicuna-7b-v1.5/tokenizer.model"
```

| 参数 | 值 | 文件位置 | 估算大小 |
|------|----|----------|----------|
| `pretrained_llm_path` | `./checkpoints/vicuna-7b-v1.5/lit_model.pth` | 硬编码 | ~13GB |
| `tokenizer_llm_path` | `./checkpoints/vicuna-7b-v1.5/tokenizer.model` | 硬编码 | ~4MB |
| `checkpoint_dir` | `checkpoints/vicuna-7b-v1.5` | 硬编码 | - |

### 3. 从options.py获取的默认参数值

#### 从CLI.py第31行加载: `args = option.get_args_parser()`

```python
# options.py 中的默认值
```

| 参数分类 | 参数名 | 默认值 | 说明 |
|----------|--------|--------|------|
| **LoRA参数** | `--lora_r` | `64` | LoRA秩 |
| | `--lora_alpha` | `16` | LoRA缩放因子 |
| | `--lora_dropout` | `0.05` | LoRA dropout率 |
| **多模态编码器** | `--video_tower` | `LanguageBind/LanguageBind_Video_merge` | 视频编码器 |
| | `--image_tower` | `LanguageBind/LanguageBind_Image` | 图像编码器 |
| | `--mm_vision_select_layer` | `-2` | 视觉编码器层选择 |
| | `--mm_projector_type` | `mlp2x_gelu` | 投影器类型 |
| | `--mm_hidden_size` | `1024` | 多模态特征维度 |
| | `--hidden_size` | `4096` | 语言模型特征维度 |
| **其他参数** | `--pretrained_llama` | `"13B"` | 预训练LLM标识 |
| | `--dataname` | `'t2m'` | 数据集名称 |
| | `--out_dir` | `'./out/'` | 输出目录 |
| | `--vqvae_pth` | `'/comp_robot/lushunlin/MotionGPT/checkpoints/pretrained_vqvae/t2m.pth'` | VQ-VAE路径 |
| | `--data_dir` | `'./data/'` | 数据目录 |
| | `--block_size` | `512` | 序列块大小 |
| | `--batch_size` | `256` | 批次大小 |
| | `--micro_batch_size` | `4` | 微批次大小 |
| | `--learning_rate_lora` | `3e-3` | LoRA学习率 |
| | `--learning_rate_mlp` | `3e-3` | MLP学习率 |
| | `--weight_decay` | `0.01` | 权重衰减 |
| | `--warmup_steps` | `100` | 预热步数 |
| | `--eval_interval` | `100` | 评估间隔 |
| | `--save_interval` | `100` | 保存间隔 |
| | `--eval_iters` | `100` | 评估迭代数 |
| | `--log_interval` | `1` | 日志间隔 |

### 4. main()函数的默认参数值

```python
# CLI.py 第194-201行
def main(
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",
) -> None:
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quantize` | `None` | 量化模式 |
| `dtype` | `"float32"` | 数据类型 |
| `max_new_tokens` | `200` | 最大生成token数 |
| `top_k` | `200` | top-k采样 |
| `temperature` | `0.8` | 采样温度 |
| `accelerator` | `"auto"` | 加速器选择 |

### 5. LoRA目标层配置（硬编码）

```python
# CLI.py 第227-232行
lora_query = True          # 应用于查询层
lora_key = False          # 不应用于键层
lora_value = True         # 应用于值层
lora_projection = False  # 不应用于投影层
lora_mlp = False          # 不应用于MLP层
lora_head = False         # 不应用于输出层
```

### 6. 模型配置参数

#### LLaMA模型配置
```python
# CLI.py 第233-244行
config = Config.from_name(
    name=checkpoint_dir.name,        # "vicuna-7b-v1.5"
    r=args.lora_r,                  # 64
    alpha=args.lora_alpha,           # 16
    dropout=args.lora_dropout,       # 0.05
    to_query=lora_query,            # True
    to_key=lora_key,                # False
    to_value=lora_value,            # True
    to_projection=lora_projection,  # False
    to_mlp=lora_mlp,                # False
    to_head=lora_head,              # False
)
```

#### 多模态配置
```python
# CLI.py 第250行
X = ['Video']  # 当前只处理视频模态

# CLI.py 第255行
model_path = 'LanguageBind/Video-LLaVA-7B'
```

### 7. 推理时的固定参数

```python
# CLI.py 第320行
max_seq_length = 4096  # 最大序列长度

# CLI.py 第258行
linear_proj = mm_backbone_mlp_model.mm_projector  # 多模态投影器

# CLI.py 第297行
tensor = video_tensor.to('cuda', dtype=torch.float16)  # 视频处理使用半精度
```

## 完整参数值汇总表

### 🎯 核心模型参数
| 参数名 | 值 | 来源 | 说明 |
|---------|-----|------|------|
| **基础LLM** | | | |
| `pretrained_llm_path` | `./checkpoints/vicuna-7b-v1.5/lit_model.pth` | 硬编码 | Vicuna-7B基础模型 |
| `tokenizer_llm_path` | `./checkpoints/vicuna-7b-v1.5/tokenizer.model` | 硬编码 | 分词器模型 |
| `checkpoint_dir` | `checkpoints/vicuna-7b-v1.5` | 硬编码 | 模型目录 |
| **LoRA配置** | | | |
| `lora_path` | `./finetuned_models/video_qa_lora.pth` | 用户指定 | LoRA微调权重 |
| `lora_r` | `64` | options.py默认 | LoRA秩 |
| `lora_alpha` | `16` | options.py默认 | LoRA缩放因子 |
| `lora_dropout` | `0.05` | options.py默认 | LoRA dropout率 |
| `lora_query` | `True` | 硬编码 | 应用于查询层 |
| `lora_key` | `False` | 硬编码 | 不应用于键层 |
| `lora_value` | `True` | 硬编码 | 应用于值层 |
| `lora_projection` | `False` | 硬编码 | 不应用于投影层 |
| `lora_mlp` | `False` | 硬编码 | 不应用于MLP层 |
| `lora_head` | `False` | 硬编码 | 不应用于输出层 |

### 🎥 多模态编码器参数
| 参数名 | 值 | 来源 | 说明 |
|---------|-----|------|------|
| **编码器配置** | | | |
| `video_tower` | `LanguageBind/LanguageBind_Video_merge` | options.py默认 | 视频编码器 |
| `image_tower` | `LanguageBind/LanguageBind_Image` | options.py默认 | 图像编码器 |
| `model_path` | `LanguageBind/Video-LLaVA-7B` | 硬编码 | 多模态模型路径 |
| `mm_vision_select_layer` | `-2` | options.py默认 | 选择倒数第二层 |
| **投影器配置** | | | |
| `mlp_path` | `./finetuned_models/video_mlp_projector.pth` | 用户指定 | 投影器权重 |
| `mm_projector_type` | `mlp2x_gelu` | options.py默认 | 投影器类型 |
| `mm_hidden_size` | `1024` | options.py默认 | 多模态特征维度 |
| `hidden_size` | `4096` | options.py默认 | 语言模型特征维度 |
| **数据维度** | | | |
| 输入视频形状 | `(1, 3, 8, 224, 224)` | 硬编码 | B×C×T×H×W |
| 输出特征形状 | `(1, 2048, 4096)` | 运行时确定 | 编码后特征 |

### ⚙️ 推理参数
| 参数名 | 值 | 来源 | 说明 |
|---------|-----|------|------|
| **生成参数** | | | |
| `max_new_tokens` | `200` | main函数默认 | 最大生成token数 |
| `top_k` | `200` | main函数默认 | top-k采样 |
| `temperature` | `0.8` | main函数默认 | 采样温度 |
| `max_seq_length` | `4096` | 硬编码 | 最大序列长度 |
| **系统参数** | | | |
| `dtype` | `"float32"` | main函数默认 | 数据类型 |
| `quantize` | `None` | main函数默认 | 量化模式 |
| `accelerator` | `"auto"` | main函数默认 | 加速器选择 |
| **精度控制** | | | |
| 模型精度 | `bfloat16` | 硬编码 | 模型运行精度 |
| 视频处理精度 | `float16` | 硬编码 | 视频编码精度 |

### 📊 训练相关参数（用于模型配置）
| 参数名 | 值 | 来源 | 说明 |
|---------|-----|------|------|
| **批次配置** | | | |
| `batch_size` | `256` | options.py默认 | 批次大小 |
| `micro_batch_size` | `4` | options.py默认 | 微批次大小 |
| `block_size` | `512` | options.py默认 | 序列块大小 |
| **学习率配置** | | | |
| `learning_rate_lora` | `3e-3` | options.py默认 | LoRA学习率 |
| `learning_rate_mlp` | `3e-3` | options.py默认 | MLP学习率 |
| `weight_decay` | `0.01` | options.py默认 | 权重衰减 |
| **训练调度** | | | |
| `warmup_steps` | `100` | options.py默认 | 预热步数 |
| `eval_interval` | `100` | options.py默认 | 评估间隔 |
| `save_interval` | `100` | options.py默认 | 保存间隔 |
| `eval_iters` | `100` | options.py默认 | 评估迭代数 |
| `log_interval` | `1` | options.py默认 | 日志间隔 |

### 🗂️ 数据和路径参数
| 参数名 | 值 | 来源 | 说明 |
|---------|-----|------|------|
| **数据集配置** | | | |
| `dataname` | `'t2m'` | options.py默认 | 数据集名称 |
| `data_dir` | `'./data/'` | options.py默认 | 数据目录 |
| `out_dir` | `'./out/'` | options.py默认 | 输出目录 |
| **检查点配置** | | | |
| `vqvae_pth` | `/comp_robot/lushunlin/MotionGPT/checkpoints/pretrained_vqvae/t2m.pth` | options.py默认 | VQ-VAE路径 |
| `resume_pth` | `None` | options.py默认 | 恢复检查点 |

## 关键数据流维度

### 1. 视频数据处理流程
```python
# 原始视频 → 预处理 → 编码 → 投影 → 语言模型
视频文件 → (1,3,8,224,224) → (1,2048,1024) → (1,2048,4096) → 与文本融合
```

### 2. 维度变化过程
```
步骤1: 视频预处理      → (1, 3, 8, 224, 224)   # 8帧×224×224 RGB
步骤2: LanguageBind编码 → (1, 2048, 1024)     # 2048个tokens, 1024维
步骤3: MLP投影器      → (1, 2048, 4096)     # 投影到语言模型空间
步骤4: 与文本融合      → (seq_len, 4096)     # 与文本token拼接
```

### 3. 内存使用估算
```python
# 基于参数的内存估算
基础模型: ~13GB (Vicuna-7B FP32)
LoRA权重: ~100MB-1GB (取决于训练数据)
MLP投影器: ~10-100MB
LanguageBind编码器: ~2-4GB (首次下载后缓存)
总计运行时内存: ~20-25GB (GPU)
```

## 实际运行时的参数获取方式

### 1. 命令行参数解析流程
```bash
# 1. 解析用户输入
python CLI.py --lora_path ./finetuned_models/video_qa_lora.pth --mlp_path ./finetuned_models/video_mlp_projector.pth

# 2. 加载默认配置 (options.py)
args = option.get_args_parser()

# 3. 应用main函数默认参数
quantize=None, dtype="float32", max_new_tokens=200, top_k=200, temperature=0.8, accelerator="auto"

# 4. 应用硬编码路径
pretrained_llm_path = "./checkpoints/vicuna-7b-v1.5/lit_model.pth"
tokenizer_llm_path = "./checkpoints/vicuna-7b-v1.5/tokenizer.model"
```

### 2. 模型加载优先级
```python
# 1. 加载基础LLM权重
pretrained_llm_checkpoint = lazy_load(pretrained_llm_path)

# 2. 加载LoRA微调权重
lora_checkpoint = lazy_load(lora_path)

# 3. 合并权重
model_state_dict = {**pretrained_llm_checkpoint, **lora_checkpoint}

# 4. 加载MLP投影器
pretrained_checkpoint_mlp = torch.load(mlp_path)
linear_proj.load_state_dict(pretrained_checkpoint_mlp)
```

这个详细的参数分析可以帮助您理解MotionLLM CLI运行时的完整配置状态，以及各个参数的来源和作用。