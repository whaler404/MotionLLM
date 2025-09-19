# MotionLLM 模型配置指南

## 概述

MotionLLM 使用统一的配置系统来控制模型架构、训练参数和多模态组件。本文档详细说明了各个配置参数的含义和推荐设置。

## 1. 多模态编码器配置

### 1.1 基础配置参数

```python
# 编码器选择
image_tower = "LanguageBind_Image"          # 图像编码器
video_tower = "LanguageBind_Video_merge"    # 视频编码器

# 特征提取配置
mm_vision_select_layer = -2                 # 选择倒数第二层特征
mm_hidden_size = 1024                       # 多模态特征维度
```

### 1.2 支持的编码器类型

#### LanguageBind 系列
- `"LanguageBind_Image"`: LanguageBind 图像编码器
- `"LanguageBind_Video_merge"`: LanguageBind 视频编码器（合并特征）
- `"LanguageBind_Audio"`: LanguageBind 音频编码器
- `"LanguageBind_Thermal"`: LanguageBind 热成像编码器
- `"LanguageBind_Depth"`: LanguageBind 深度编码器

#### CLIP 系列
- `"openai/clip-vit-base-patch32"`: OpenAI CLIP Base
- `"openai/clip-vit-large-patch14"`: OpenAI CLIP Large
- `"laion/CLIP-ViT-B-32-laion2B-s34B-b79K"`: LAION CLIP

#### MAE 系列
- 包含 `"mae"` 字符串的模型名称

### 1.3 配置示例

```python
# 推荐配置 - LanguageBind
config.image_tower = "LanguageBind_Image"
config.video_tower = "LanguageBind_Video_merge"
config.mm_vision_select_layer = -2
config.mm_hidden_size = 1024

# 推荐配置 - CLIP
config.image_tower = "openai/clip-vit-large-patch14"
config.video_tower = "openai/clip-vit-large-patch14"
config.mm_vision_select_layer = -2
config.mm_hidden_size = 1024
```

## 2. 多模态投影器配置

### 2.1 投影器类型配置

```python
mm_projector_type = "mlp2x_gelu"  # 默认投影器类型
```

### 2.2 支持的投影器类型

#### 线性投影器
- `"linear"`: 单层线性投影
  ```python
  nn.Linear(mm_hidden_size, hidden_size)
  ```

#### MLP 投影器
- `"mlp1x_gelu"`: 1 层 MLP
  ```python
  nn.Linear(mm_hidden_size, hidden_size)
  ```

- `"mlp2x_gelu"`: 2 层 MLP（推荐）
  ```python
  nn.Sequential(
      nn.Linear(1024, 4096),
      nn.GELU(),
      nn.Linear(4096, 4096)
  )
  ```

- `"mlp3x_gelu"`: 3 层 MLP
  ```python
  nn.Sequential(
      nn.Linear(1024, 4096),
      nn.GELU(),
      nn.Linear(4096, 4096),
      nn.GELU(),
      nn.Linear(4096, 4096)
  )
  ```

#### Q-Former 投影器
- `"qformer2_64"`: 2 层 Q-Former，64 个查询
- `"qformer4_32"`: 4 层 Q-Former，32 个查询
- `"qformer6_64"`: 6 层 Q-Former，64 个查询

#### 恒等映射
- `"identity"`: 直接传递特征（不推荐用于生产）

### 2.3 配置建议

```python
# 根据模型复杂度选择
config.mm_projector_type = "mlp2x_gelu"  # 平衡性能和效果
config.hidden_size = 4096               # Vicuna 隐藏层维度
config.mm_hidden_size = 1024            # 编码器输出维度
```

## 3. VQ-VAE 配置

### 3.1 基础架构配置

```python
# Codebook 配置
code_dim = 512                          # 编码维度
nb_code = 1024                         # codebook 大小
quantizer = "ema_reset"                 # 量化器类型

# 网络架构配置
output_emb_width = 512                  # 输出嵌入宽度
down_t = 3                              # 下采样次数
stride_t = 2                            # 下采样步长
width = 512                             # 网络宽度
depth = 3                               # 网络深度
dilation_growth_rate = 3                # 膨胀增长率
```

### 3.2 量化器类型

#### EMA 系列
- `"ema"`: 标准 EMA 更新
  ```python
  mu = 0.99  # 固定动量
  ```

- `"ema_reset"`: EMA + 重置机制（推荐）
  ```python
  mu = args.mu  # 可配置动量
  ```

#### 其他类型
- `"orig"`: 原始 VQ-VAE 量化器
- `"reset"`: 重置机制量化器

### 3.3 激活函数和归一化

```python
activation = 'relu'                     # 激活函数类型
norm = None                             # 归一化类型

# 支持的激活函数
activation_options = ['relu', 'silu', 'gelu']

# 支持的归一化
norm_options = [None, 'LN', 'GN', 'BN']
```

### 3.4 数据集特定配置

```python
# 动作维度配置
if dataname == 'kit':
    input_features = 251     # KIT 数据集
    nb_joints = 21           # 21 个关节
elif dataname == 't2m':
    input_features = 263     # HumanML3D 数据集
    nb_joints = 22           # 22 个关节
```

### 3.5 推荐配置

```python
# 基础配置
config.code_dim = 512
config.nb_code = 1024
config.quantizer = "ema_reset"
config.output_emb_width = 512
config.down_t = 3
config.stride_t = 2
config.width = 512
config.depth = 3
config.dilation_growth_rate = 3
config.activation = 'relu'
config.norm = None

# 高级配置
config.mu = 0.99                           # EMA 动量
```

## 4. 训练配置

### 4.1 LoRA 配置

```python
# LoRA 基础配置
lora_path = "path/to/lora/checkpoint.pth"  # LoRA 检查点路径
lora_r = 64                                # LoRA 秩
lora_alpha = 16                            # LoRA 缩放因子
lora_dropout = 0.05                        # LoRA Dropout
```

### 4.2 优化器配置

```python
# 学习率配置
learning_rate_lora = 2e-5                  # LoRA 学习率
learning_rate_mlp = 1e-3                   # MLP 投影器学习率

# 优化器参数
weight_decay = 0.0                         # 权重衰减
warmup_steps = 1000                        # 预热步数
```

### 4.3 训练调度

```python
# 批处理配置
batch_size = 32                            # 总批次大小
micro_batch_size = 4                       # 微批次大小
gradient_accumulation_steps = batch_size // micro_batch_size

# 评估和保存配置
eval_interval = 100                        # 评估间隔
save_interval = 500                        # 保存间隔
eval_iters = 100                           # 评估迭代数
log_interval = 10                          # 日志间隔
```

### 4.4 推荐训练配置

```python
# 推荐配置
config.lora_r = 64
config.lora_alpha = 16
config.lora_dropout = 0.05
config.learning_rate_lora = 2e-5
config.learning_rate_mlp = 1e-3
config.weight_decay = 0.0
config.warmup_steps = 1000
config.batch_size = 32
config.micro_batch_size = 4
config.eval_interval = 100
config.save_interval = 500
```

## 5. 评估配置

### 5.1 数据集配置

```python
# 数据集选择
dataset_name = "t2m"                       # 或 "kit"

# 数据集特定参数
if dataset_name == 't2m':
    dim_pose = 263
    max_motion_length = 196
elif dataset_name == 'kit':
    dim_pose = 251
    max_motion_length = 196
```

### 5.2 评估模型配置

```python
# 评估模型维度
dim_word = 300                            # 词向量维度
dim_pos_ohot = len(POS_enumerator)        # 位置编码维度
dim_motion_hidden = 1024                  # 运动隐藏层维度
max_text_len = 20                         # 最大文本长度
dim_text_hidden = 512                     # 文本隐藏层维度
dim_coemb_hidden = 512                    # 联合嵌入维度
dim_movement_enc_hidden = 512             # 运动编码隐藏层维度
dim_movement_latent = 256                 # 运动潜在维度
```

### 5.3 评估路径配置

```python
# 检查点路径
checkpoints_dir = "path/to/checkpoints"
motion_vq_token_path = "path/to/vq/tokens"
motionx_zero_shot_path = "path/to/zero/shot/data"
```

## 6. 推理配置

### 6.1 模型加载配置

```python
# 基础模型路径
resume_pth = "path/to/base/model.pth"     # 基础模型路径
vqvae_pth = "path/to/vqvae/model.pth"      # VQ-VAE 模型路径

# LoRA 和投影器路径
lora_path = "path/to/lora/adapter.pth"    # LoRA 适配器路径
mlp_path = "path/to/mlp/projector.pth"    # MLP 投影器路径
```

### 6.2 生成配置

```python
# 文本配置
block_size = 2048                         # 上下文窗口大小

# 生成策略
temperature = 0.7                         # 生成温度
top_p = 0.9                               # Top-p 采样
max_new_tokens = 512                      # 最大生成长度
```

### 6.3 特殊功能配置

```python
# 生成策略
diverse = False                           # 多样化生成
projectionnn = False                      # 投影网络
vinilla = False                           # 基础版本
```

## 7. 完整配置示例

### 7.1 标准配置

```python
# configs/standard_config.py
class StandardConfig:
    # 多模态编码器
    image_tower = "LanguageBind_Image"
    video_tower = "LanguageBind_Video_merge"
    mm_vision_select_layer = -2
    mm_hidden_size = 1024
    
    # 投影器
    mm_projector_type = "mlp2x_gelu"
    hidden_size = 4096
    
    # VQ-VAE
    code_dim = 512
    nb_code = 1024
    quantizer = "ema_reset"
    output_emb_width = 512
    down_t = 3
    stride_t = 2
    width = 512
    depth = 3
    dilation_growth_rate = 3
    activation = 'relu'
    norm = None
    
    # LoRA
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.05
    
    # 训练
    learning_rate_lora = 2e-5
    learning_rate_mlp = 1e-3
    weight_decay = 0.0
    warmup_steps = 1000
    batch_size = 32
    micro_batch_size = 4
    
    # 评估
    dataset_name = "t2m"
    dim_pose = 263
    max_motion_length = 196
```

### 7.2 高级配置

```python
# configs/advanced_config.py
class AdvancedConfig:
    # 多模态编码器
    image_tower = "LanguageBind_Image"
    video_tower = "LanguageBind_Video_merge"
    mm_vision_select_layer = -2
    mm_hidden_size = 1024
    
    # 投影器
    mm_projector_type = "qformer4_64"
    hidden_size = 4096
    
    # VQ-VAE
    code_dim = 512
    nb_code = 2048  # 更大的 codebook
    quantizer = "ema_reset"
    output_emb_width = 512
    down_t = 4  # 更深的下采样
    stride_t = 2
    width = 1024  # 更宽的网络
    depth = 4  # 更深的网络
    dilation_growth_rate = 2
    activation = 'gelu'
    norm = 'LN'
    
    # LoRA
    lora_r = 128  # 更高秩
    lora_alpha = 32
    lora_dropout = 0.1
    
    # 训练
    learning_rate_lora = 1e-5  # 更低的学习率
    learning_rate_mlp = 5e-4
    weight_decay = 0.01
    warmup_steps = 2000
    batch_size = 16  # 更大的批次
    micro_batch_size = 2
```

## 8. 配置验证

### 8.1 配置检查函数

```python
def validate_config(config):
    """验证配置参数的有效性"""
    
    # 检查维度匹配
    assert config.mm_hidden_size > 0
    assert config.hidden_size > 0
    assert config.mm_hidden_size <= config.hidden_size
    
    # 检查投影器类型
    valid_projectors = ['linear', 'mlp1x_gelu', 'mlp2x_gelu', 'mlp3x_gelu', 
                       'qformer2_64', 'qformer4_64', 'qformer6_64', 'identity']
    assert config.mm_projector_type in valid_projectors
    
    # 检查量化器类型
    valid_quantizers = ['ema', 'ema_reset', 'orig', 'reset']
    assert config.quantizer in valid_quantizers
    
    # 检查数据集
    assert config.dataname in ['t2m', 'kit']
    
    # 检查学习率
    assert config.learning_rate_lora > 0
    assert config.learning_rate_mlp > 0
    
    return True
```

### 8.2 配置优化建议

```python
def optimize_config(config):
    """根据硬件条件优化配置"""
    
    # GPU 内存优化
    if torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:
        config.batch_size = min(config.batch_size, 16)
        config.micro_batch_size = min(config.micro_batch_size, 2)
    
    # 多 GPU 优化
    if torch.cuda.device_count() > 1:
        config.batch_size = config.batch_size * torch.cuda.device_count()
    
    return config
```

这个配置指南提供了 MotionLLM 的完整配置说明，帮助用户根据具体需求调整模型参数。