# MotionLLM 模型配置详细说明

## 模型加载地址和检查点配置

### 1. 核心模型文件

#### 基础语言模型 (Vicuna-7B)
```python
# 代码中的硬编码路径
pretrained_llm_path = "./checkpoints/vicuna-7b-v1.5/lit_model.pth"
tokenizer_llm_path = "./checkpoints/vicuna-7b-v1.5/tokenizer.model"

# 文件结构
checkpoints/
└── vicuna-7b-v1.5/
    ├── lit_model.pth          # 预训练权重 (~13GB)
    ├── tokenizer.model        # 分词器模型 (~4MB)
    └── config.json            # 模型配置文件
```

#### LoRA 微调模型
```bash
# 命令行参数
--lora_path /path/to/your/lora_adapter.pth

# 文件结构
your_finetuned_models/
├── lora_adapter.pth          # LoRA权重 (~100MB-1GB)
├── training_config.json      # 训练配置
└── evaluation_results.json   # 评估结果
```

#### 多模态投影器
```bash
# 命令行参数
--mlp_path /path/to/your/mlp_projector.pth

# 文件结构
your_finetuned_models/
├── mlp_projector.pth         # 投影器权重 (~10-100MB)
├── projector_config.json     # 投影器配置
└── training_logs/            # 训练日志
```

### 2. 多模态编码器配置

#### LanguageBind 编码器
```python
# 默认配置（来自 options.py）
image_tower = "LanguageBind/LanguageBind_Image"
video_tower = "LanguageBind/LanguageBind_Video_merge"

# HuggingFace 模型ID
LanguageBind/LanguageBind_Image           # 图像编码器
LanguageBind/LanguageBind_Video_merge      # 视频编码器
LanguageBind/LanguageBind_Audio           # 音频编码器
LanguageBind/LanguageBind_Thermal          # 热成像编码器
LanguageBind/LanguageBind_Depth            # 深度编码器

# 缓存目录结构
cache_dir/
├── LanguageBind_Image/
│   ├── pytorch_model.bin      # 预训练权重
│   ├── config.json            # 模型配置
│   └── preprocessor_config.json # 预处理器配置
└── LanguageBind_Video_merge/
    ├── pytorch_model.bin      # 预训练权重
    ├── config.json            # 模型配置
    └── preprocessor_config.json # 预处理器配置
```

### 3. VQ-VAE 运动编码器

#### 预训练 VQ-VAE 模型
```bash
# 命令行参数
--vqvae_pth /path/to/pretrained_vqvae.pth

# 默认路径（来自 options.py）
default_vqvae_pth = "/comp_robot/lushunlin/MotionGPT/checkpoints/pretrained_vqvae/t2m.pth"

# 文件结构
checkpoints/
└── pretrained_vqvae/
    ├── t2m.pth                 # HumanML3D 数据集训练的 VQ-VAE
    ├── kit.pth                 # KIT 数据集训练的 VQ-VAE
    └── vqvae_config.json       # VQ-VAE 配置文件
```

## 详细参数配置说明

### 1. 模型架构参数

#### LoRA 配置
```python
# LoRA 微调参数
parser.add_argument('--lora_r', type=int, default=64)                    # LoRA 秩
parser.add_argument('--lora_alpha', type=int, default=16)                # LoRA 缩放因子
parser.add_argument('--lora_dropout', type=float, default=0.05)          # LoRA dropout率

# LoRA 目标层配置（代码中硬编码）
lora_query = True          # 应用于查询层
lora_key = False          # 不应用于键层
lora_value = True         # 应用于值层
lora_projection = False  # 不应用于投影层
lora_mlp = False          # 不应用于MLP层
lora_head = False         # 不应用于输出层
```

#### 多模态投影器配置
```python
# 投影器类型和维度
parser.add_argument('--mm_projector_type', type=str, default='mlp2x_gelu')  # 投影器类型
parser.add_argument('--mm_hidden_size', type=int, default=1024)           # 多模态特征维度
parser.add_argument('--hidden_size', type=int, default=4096)              # 语言模型维度
parser.add_argument('--mm_vision_select_layer', type=int, default=-2)    # 视觉编码器层选择

# 支持的投影器类型
# 'linear'                    # 线性投影
# 'mlp2x_gelu'                 # 2层MLP + GELU
# 'mlp3x_gelu'                 # 3层MLP + GELU
# 'qformer2_64'                # Q-Former (2层, 64查询tokens)
# 'identity'                   # 恒等映射
```

#### 视觉编码器配置
```python
# 编码器选择
parser.add_argument('--image_tower', type=str, default='LanguageBind/LanguageBind_Image')
parser.add_argument('--video_tower', type=str, default='LanguageBind/LanguageBind_Video_merge')

# 特征选择
parser.add_argument('--mm_vision_select_layer', type=int, default=-2)     # -2表示倒数第二层
parser.add_argument('--mm_vision_select_feature', type=str, default='patch')  # 'patch'或'cls_patch'
```

### 2. 训练参数配置

#### 基础训练参数
```python
# 数据加载和批次
parser.add_argument('--batch_size', type=int, default=256)                 # 批次大小
parser.add_argument('--micro_batch_size', type=int, default=4)           # 微批次大小
parser.add_argument('--block_size', type=int, default=512)               # 序列块大小

# 学习率和优化
parser.add_argument('--learning_rate_lora', type=float, default=3e-3)     # LoRA学习率
parser.add_argument('--learning_rate_mlp', type=float, default=3e-3)      # MLP学习率
parser.add_argument('--weight_decay', type=float, default=0.01)           # 权重衰减

# 训练调度
parser.add_argument('--warmup_steps', type=int, default=100)              # 预热步数
parser.add_argument('--eval_interval', type=int, default=100)             # 评估间隔
parser.add_argument('--save_interval', type=int, default=100)            # 保存间隔
parser.add_argument('--eval_iters', type=int, default=100)               # 评估迭代数
parser.add_argument('--log_interval', type=int, default=1)               # 日志间隔
```

#### VQ-VAE 参数
```python
# VQ-VAE 架构
parser.add_argument("--code_dim", type=int, default=512)                  # 编码维度
parser.add_argument("--nb_code", type=int, default=512)                   # 编码数量
parser.add_argument("--mu", type=float, default=0.99)                     # 指数移动平均
parser.add_argument("--down_t", type=int, default=2)                     # 下采样率
parser.add_argument("--stride_t", type=int, default=2)                   # 步长
parser.add_argument("--width", type=int, default=512)                    # 网络宽度
parser.add_argument("--depth", type=int, default=3)                      # 网络深度
parser.add_argument("--dilation_growth_rate", type=int, default=3)       # 膨胀率
parser.add_argument("--output_emb_width", type=int, default=512)          # 输出嵌入宽度
parser.add_argument('--vq_act', type=str, default='relu')                 # 激活函数
parser.add_argument("--window_size", type=int, default=64)                # 窗口大小
```

#### 量化器参数
```python
# 量化器配置
parser.add_argument("--quantizer", type=str, default='ema_reset')          # 量化器类型
parser.add_argument('--quantbeta', type=float, default=1.0)              # 量化beta值

# 支持的量化器类型
# 'ema'           # 指数移动平均
# 'orig'          # 原始量化器
# 'ema_reset'     # 带重置的EMA
# 'reset'         # 重置量化器
```

### 3. 推理参数配置

#### 生成参数
```python
# CLI.py 中的推理参数（函数参数）
quantize: Optional[str] = None          # 量化模式
dtype: str = "float32"                  # 数据类型
max_new_tokens: int = 200               # 最大生成token数
top_k: int = 200                        # top-k采样
temperature: float = 0.8                # 采样温度
accelerator: str = "auto"               # 加速器选择
```

#### 任务和模式参数
```python
# 任务描述和输入
parser.add_argument('--prompt', type=str, 
    default="Generate a textual description corresponding to the given sequence of human motion tokens.")
parser.add_argument('--input', type=str, help='generation conditions')

# 数据集和输出
parser.add_argument('--dataname', type=str, default='t2m')               # 数据集名称
parser.add_argument('--data_dir', type=str, default='./data/')           # 数据目录
parser.add_argument('--out_dir', type=str, default='./out/')             # 输出目录

# 模型恢复和检查点
parser.add_argument('--vqvae_pth', type=str, help='path to the pretrained vqvae pth')
parser.add_argument('--resume_pth', type=str, help='path to saved finetuned model')
parser.add_argument('--lora_path', type=str, help='path to fintuned model for evaluation')
parser.add_argument('--mlp_path', type=str, help='mlp path')
```

### 4. 特殊功能参数

#### 可视化和评估
```python
# 渲染和可视化
parser.add_argument("--render", action='store_true', help='render smpl')
parser.add_argument("--motion_vq_token_path", type=str, help='vq token path for motion visualization')

# 零样本评估
parser.add_argument('--motionx_zero_shot_path', type=str, help='zero shot motion dataset directory')
```

#### 实验配置
```python
# 特殊模式
parser.add_argument("--projectionnn", action='store_true', help='MLP projection')
parser.add_argument("--diverse", action='store_true', help='diverse description')
parser.add_argument("--vinilla", action='store_true', help='vinilla motion')

# 模型类型标识
parser.add_argument('--model_type', type=str, default=None, help='if use multimodal video tower')
```

## 模型维度配置指南

### 1. 维度一致性检查

#### 常见维度配置
```python
# 维度映射关系
Model -> Hidden Size
├── CLIP-BASE -> 768
├── LanguageBind -> 1024
├── MAE-BASE -> 768
├── Vicuna-7B -> 4096
└── LLaMA-13B -> 5120

# 投影器配置示例
config = {
    'mm_hidden_size': 1024,      # LanguageBind特征维度
    'hidden_size': 4096,         # Vicuna-7B特征维度
    'mm_projector_type': 'mlp2x_gelu'  # 投影器类型
}
```

#### 维度验证
```python
def validate_model_dimensions(config):
    """验证模型维度一致性"""
    
    # 检查投影器输入输出维度
    if config.mm_hidden_size > config.hidden_size:
        print(f"Warning: Input dimension ({config.mm_hidden_size}) > Output dimension ({config.hidden_size})")
    
    # 检查LoRA配置
    if hasattr(config, 'lora_r') and config.lora_r > 128:
        print(f"Warning: Large LoRA rank ({config.lora_r}) may cause memory issues")
    
    # 检查序列长度
    if hasattr(config, 'block_size') and config.block_size > 2048:
        print(f"Warning: Large block size ({config.block_size}) may cause memory issues")
```

### 2. 内存使用估算

#### 模型大小估算
```python
def estimate_model_memory_usage(config):
    """估算模型内存使用"""
    
    # 基础模型大小
    base_model_size = 13  # Vicuna-7B ~13GB (FP32)
    
    # LoRA参数量
    lora_params = config.lora_r * config.lora_alpha * 4  # 近似计算
    lora_size = lora_params * 4 / (1024**3)  # GB
    
    # 投影器参数量
    if config.mm_projector_type == 'linear':
        proj_params = config.mm_hidden_size * config.hidden_size
    elif config.mm_projector_type == 'mlp2x_gelu':
        proj_params = config.mm_hidden_size * config.hidden_size * 2
    elif 'qformer' in config.mm_projector_type:
        proj_params = config.mm_hidden_size * config.hidden_size * 3
    
    proj_size = proj_params * 4 / (1024**3)  # GB
    
    # 总内存需求
    total_memory = base_model_size + lora_size + proj_size
    
    print(f"模型内存使用估算:")
    print(f"  基础模型: {base_model_size:.1f} GB")
    print(f"  LoRA适配器: {lora_size:.2f} GB")
    print(f"  投影器: {proj_size:.2f} GB")
    print(f"  总计: {total_memory:.1f} GB")
    
    return total_memory
```

### 3. 性能优化配置

#### 内存优化配置
```python
# 低内存配置
low_memory_config = {
    'lora_r': 32,                    # 减少LoRA秩
    'lora_alpha': 8,                 # 减少LoRA缩放因子
    'micro_batch_size': 1,           # 减少微批次大小
    'block_size': 256,               # 减少序列长度
    'mm_projector_type': 'linear',    # 使用线性投影器
}

# 高性能配置
high_performance_config = {
    'lora_r': 128,                   # 增加LoRA秩
    'lora_alpha': 32,                # 增加LoRA缩放因子
    'micro_batch_size': 8,           # 增加微批次大小
    'block_size': 1024,              # 增加序列长度
    'mm_projector_type': 'mlp3x_gelu', # 使用3层MLP
}

# 平衡配置
balanced_config = {
    'lora_r': 64,                    # 中等LoRA秩
    'lora_alpha': 16,                # 中等LoRA缩放因子
    'micro_batch_size': 4,           # 中等微批次大小
    'block_size': 512,               # 中等序列长度
    'mm_projector_type': 'mlp2x_gelu', # 使用2层MLP
}
```

#### 训练稳定性配置
```python
# 稳定训练配置
stable_training_config = {
    'learning_rate_lora': 1e-4,      # 较低的学习率
    'learning_rate_mlp': 1e-4,       # 较低的学习率
    'weight_decay': 0.01,            # 适中的权重衰减
    'warmup_steps': 200,             # 增加预热步数
    'lora_dropout': 0.1,             # 增加dropout
    'batch_size': 128,               # 适中的批次大小
}
```

## 配置文件示例

### 1. 完整配置示例
```json
{
    "model": {
        "pretrained_llm_path": "./checkpoints/vicuna-7b-v1.5/lit_model.pth",
        "tokenizer_llm_path": "./checkpoints/vicuna-7b-v1.5/tokenizer.model",
        "lora_path": "./finetuned_models/video_qa_lora.pth",
        "mlp_path": "./finetuned_models/video_mlp_projector.pth"
    },
    "multimodal": {
        "image_tower": "LanguageBind/LanguageBind_Image",
        "video_tower": "LanguageBind/LanguageBind_Video_merge",
        "mm_vision_select_layer": -2,
        "mm_vision_select_feature": "patch",
        "mm_projector_type": "mlp2x_gelu",
        "mm_hidden_size": 1024,
        "hidden_size": 4096
    },
    "lora": {
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    },
    "training": {
        "batch_size": 256,
        "micro_batch_size": 4,
        "block_size": 512,
        "learning_rate_lora": 3e-3,
        "learning_rate_mlp": 3e-3,
        "weight_decay": 0.01,
        "warmup_steps": 100
    },
    "vqvae": {
        "vqvae_pth": "./checkpoints/pretrained_vqvae/t2m.pth",
        "code_dim": 512,
        "nb_code": 512,
        "window_size": 64
    }
}
```

### 2. 环境配置示例
```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Python 脚本调用示例
python CLI.py \
    --lora_path ./finetuned_models/video_qa_lora.pth \
    --mlp_path ./finetuned_models/video_mlp_projector.pth \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 1024 \
    --hidden_size 4096 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 200
```

通过这些详细的配置说明，您可以更好地理解和配置MotionLLM的各个组件，以适应不同的使用场景和硬件条件。