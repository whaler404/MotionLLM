# MotionLLM 模型架构文档

## 概览

MotionLLM 是一个多模态大型语言模型，结合了文本、视频和动作处理能力。该模型基于 Vicuna 7B LLM 构建，通过 LoRA 适配器和多模态投影器实现了视频理解和动作生成功能。

## 核心模块架构

### 1. 多模态编码器 (`models/multimodal_encoder/`)

#### 1.1 架构设计
多模态编码器负责将视频和图像输入转换为特征表示，支持多种编码器类型：

- **CLIP 编码器** (`clip_encoder.py`): 标准的 CLIP 视觉编码器
- **LanguageBind 编码器** (`languagebind/`): 统一的多模态编码器框架
- **MAE 编码器** (`mae_encoder.py`): 掩码自编码器

#### 1.2 核心组件

**Builder 模式** (`builder.py`):
```python
def build_image_tower(image_tower_cfg, **kwargs)
def build_video_tower(video_tower_cfg, **kwargs)
```

**LanguageBind 架构**:
- 支持图像、视频、音频、热成像、深度图像五种模态
- 统一的编码器接口和配置系统
- 自动缓存机制和模型加载

**关键特性**:
- 视频编码器输出形状: `(batch, frames * patches, hidden)`
- 默认冻结梯度以减少推理显存
- 支持延迟加载机制

### 2. 多模态投影器 (`models/multimodal_projector/`)

#### 2.1 投影器类型
多模态投影器将编码器特征映射到 LLM 的隐空间维度：

- **线性投影器** (`linear`): 单层线性变换
- **MLP 投影器** (`mlp*dx_gelu`): 多层感知机 + GELU 激活
- **恒等映射** (`identity`): 直接传递特征
- **Q-Former 投影器** (`qformer*`): 基于 BLIP-2 的可学习查询机制

#### 2.2 实现细节

**默认配置** (`mlp2x_gelu`):
```python
modules = [
    nn.Linear(1024, 4096),  # LanguageBind -> Vicuna
    nn.GELU(),
    nn.Linear(4096, 4096)
]
```

**Q-Former 架构**:
- 可配置的查询数量和层数
- 交叉注意力机制
- 两层线性投影输出

### 3. 动作处理模块 (`models/vqvae.py`)

#### 3.1 VQ-VAE 架构
动作序列的离散化表示学习：

**VQVAE_251 类**:
- 编码器-解码器架构
- 多种量化策略 (EMA, Reset, Original)
- 支持 KIT (251维) 和 HumanML3D (263维) 数据集

**HumanVQVAE 类**:
- 封装了人体动作的特定处理
- 关节数量自适应 (KIT: 21, HumanML3D: 22)

#### 3.2 量化器实现

**量化器类型**:
- `QuantizeEMAReset`: EMA + 重置机制
- `QuantizeEMA`: 标准 EMA 更新
- `QuantizeReset`: 重置机制
- `Quantizer`: 原始 VQ-VAE 量化器

**核心算法**:
```python
# 量化过程
distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, codebook.t()) + torch.sum(codebook.t() ** 2, dim=0, keepdim=True)
code_idx = torch.min(distance, dim=-1)
```

### 4. 网络组件 (`models/encdec.py`, `models/resnet.py`)

#### 4.1 编码器-解码器架构

**Encoder 类**:
```python
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3, stride_t=2, width=512, depth=3)
```

**Decoder 类**:
- 对称的上采样架构
- 反向膨胀率配置
- 最近邻上采样 + 卷积

#### 4.2 ResNet1D 组件

**ResConv1DBlock**:
- 支持 LayerNorm、GroupNorm、BatchNorm
- 多种激活函数 (ReLU, SiLU, GELU)
- 膨胀卷积支持

**Resnet1D**:
- 可配置深度和膨胀增长率
- 支持反向膨胀率排序

### 5. 运动编码器 (`models/modules.py`)

#### 5.1 运动卷积编码器

**MovementConvEncoder**:
```python
class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size)
```

**架构特点**:
- 1D 卷积网络
- Dropout 和 LeakyReLU 激活
- Xavier 初始化

#### 5.2 双向 GRU 编码器

**TextEncoderBiGRUCo**:
- 词嵌入 + 位置编码
- 双向 GRU 处理变长序列
- 打包序列处理

**MotionEncoderBiGRUCo**:
- 运动序列编码
- 支持变长输入
- 类似文本编码器的架构

### 6. 评估包装器 (`models/evaluator_wrapper.py`)

#### 6.1 EvaluatorModelWrapper 类

**功能**:
- 文本-动作嵌入对齐评估
- 预训练模型加载
- 批量推理支持

**模型组件**:
- 运动编码器: 提取运动特征
- 文本编码器: 处理文本描述
- 运动编码器: 生成嵌入表示

## 模块间耦合关系

### 1. 数据流架构

```
输入视频/图像 → 多模态编码器 → 多模态投影器 → Vicuna LLM
输入动作 → VQ-VAE编码器 → 量化器 → 动作token → LLM集成
输入文本 → 分词器 → 文本嵌入 → LLM
```

### 2. 关键接口

**编码器接口**:
```python
# 多模态编码器
features = image_tower(pixel_values)
features = video_tower(pixel_values)

# 动作编码器
motion_tokens = vqvae.encode(motion_data)
motion_features = vqvae.encode_x(motion_data)
```

**投影器接口**:
```python
# 特征投影
projected_features = vision_projector(features)
```

### 3. 配置依赖

**关键配置参数**:
- `mm_hidden_size`: 多模态编码器输出维度
- `hidden_size`: LLM 隐空间维度
- `mm_projector_type`: 投影器类型
- `quantizer`: 量化器类型

### 4. 训练时的模块交互

**LoRA 适配**:
- 仅微调注意力层的 Query/Value 投影
- 保持 Vicuna 主干冻结

**多模态训练**:
- 编码器冻结，仅训练投影器
- VQ-VAE 可选联合训练

## 关键实现细节

### 1. 特征形状处理

**视频特征**:
- 编码器输出: `(batch, frames, patches, hidden)`
- 重塑后: `(batch, frames * patches, hidden)`
- 投影后: `(batch, frames * patches, hidden_size)`

**动作特征**:
- 输入: `(batch, timesteps, joints * 3)`
- 编码: `(batch, joints * 3, timesteps)`
- 量化: `(batch, timesteps, code_dim)`

### 2. 序列处理

**变长序列**:
- 使用 `pack_padded_sequence` 处理
- 支持 GRU 和 Transformer 架构
- 动态批处理机制

### 3. 损失计算

**VQ-VAE 损失**:
```python
commit_loss = F.mse_loss(x, x_d.detach())
reconstruction_loss = F.mse_loss(x_out, x_target)
perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
```

**投影器损失**:
- 监督微调的交叉熵损失
- 可选的对比学习损失

## 扩展性设计

### 1. 模态扩展
- 通过 `constants.py` 定义新的模态标记
- 统一的编码器接口支持新模态
- 可配置的投影器类型

### 2. 架构扩展
- 支持不同的 LLM 主干
- 可插拔的编码器组件
- 灵活的量化策略

### 3. 训练策略
- 分阶段训练支持
- 不同学习率配置
- 梯度累积和混合精度

## 性能优化

### 1. 内存优化
- 梯度检查点技术
- 混合精度训练
- 梯度累积支持

### 2. 计算优化
- 编码器冻结减少计算
- 批处理矩阵运算
- 缓存机制加速加载

### 3. 推理优化
- 模型量化支持
- 批处理推理
- 动态形状处理