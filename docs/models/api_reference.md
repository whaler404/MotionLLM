# MotionLLM API 参考文档

## 概述

本文档提供了 MotionLLM 模型各模块的详细 API 接口说明，包括初始化参数、方法签名和返回值。

## 1. 多模态编码器 API

### 1.1 CLIPVisionTower

#### 类定义
```python
class CLIPVisionTower(nn.Module)
```

#### 初始化参数
```python
def __init__(self, vision_tower, args, delay_load=False):
    """
    Args:
        vision_tower (str): CLIP 模型名称或路径
        args (Namespace): 配置参数，包含 mm_vision_select_layer 等
        delay_load (bool): 是否延迟加载模型
    """
```

#### 方法
```python
def forward(self, images):
    """
    前向传播
    
    Args:
        images (torch.Tensor): 图像张量，形状 (batch, channels, height, width)
    
    Returns:
        torch.Tensor: 图像特征，形状 (batch, sequence_length, hidden_size)
    """

def feature_select(self, image_forward_outs):
    """
    选择指定层的特征
    
    Args:
        image_forward_outs: CLIP 模型输出
    
    Returns:
        torch.Tensor: 选择的特征
    """

def load_model(self):
    """
    加载 CLIP 模型
    """
```

#### 属性
- `vision_tower_name` (str): 模型名称
- `select_layer` (int): 选择的特征层
- `is_loaded` (bool): 模型是否已加载

### 1.2 LanguageBindVideoTower

#### 类定义
```python
class LanguageBindVideoTower(nn.Module)
```

#### 初始化参数
```python
def __init__(self, video_tower, args=None, delay_load=False, **kwargs):
    """
    Args:
        video_tower (str): LanguageBind Video 模型名称
        args (Namespace): 配置参数
        delay_load (bool): 是否延迟加载
        **kwargs: 其他参数
    """
```

#### 方法
```python
def forward(self, videos):
    """
    视频编码
    
    Args:
        videos (torch.Tensor): 视频张量，形状 (batch, channels, frames, height, width)
    
    Returns:
        torch.Tensor: 视频特征，形状 (batch, frames * sequence_length, hidden_size)
    """
```

### 1.3 Builder 函数

#### build_image_tower
```python
def build_image_tower(image_tower_cfg, **kwargs):
    """
    构建图像编码器
    
    Args:
        image_tower_cfg: 配置对象，包含 image_tower 属性
        **kwargs: 其他参数
    
    Returns:
        nn.Module: 图像编码器实例
    
    Raises:
        ValueError: 未知的编码器类型
    """
```

#### build_video_tower
```python
def build_video_tower(video_tower_cfg, **kwargs):
    """
    构建视频编码器
    
    Args:
        video_tower_cfg: 配置对象，包含 video_tower 属性
        **kwargs: 其他参数
    
    Returns:
        nn.Module: 视频编码器实例
    
    Raises:
        ValueError: 未知的编码器类型
    """
```

## 2. 多模态投影器 API

### 2.1 build_vision_projector

#### 函数签名
```python
def build_vision_projector(config, delay_load=False, **kwargs):
    """
    构建视觉投影器
    
    Args:
        config: 配置对象，必须包含以下属性:
            - mm_projector_type (str): 投影器类型
            - mm_hidden_size (int): 多模态特征维度
            - hidden_size (int): LLM 隐藏层维度
        delay_load (bool): 是否延迟加载
        **kwargs: 其他参数
    
    Returns:
        nn.Module: 投影器实例
    
    Raises:
        ValueError: 未知的投影器类型
    """
```

#### 支持的投影器类型
- `'linear'`: 单层线性投影
- `'mlp*dx_gelu'`: 多层感知机（d 为层数）
- `'identity'`: 恒等映射
- `'qformer*_*'`: Q-Former 投影器（* 分别为层数和查询数）

### 2.2 Blip2Model

#### 类定义
```python
class Blip2Model(Blip2PreTrainedModel)
```

#### 初始化参数
```python
def __init__(self, config: Blip2Config):
    """
    Args:
        config (Blip2Config): BLIP-2 配置
    """
```

#### 方法
```python
def forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
    """
    Q-Former 前向传播
    
    Args:
        pixel_values (torch.Tensor): 输入特征
        output_attentions (bool, optional): 是否输出注意力
        output_hidden_states (bool, optional): 是否输出隐藏状态
        return_dict (bool, optional): 是否返回字典格式
    
    Returns:
        torch.Tensor: 查询特征
    """
```

## 3. VQ-VAE API

### 3.1 HumanVQVAE

#### 类定义
```python
class HumanVQVAE(nn.Module)
```

#### 初始化参数
```python
def __init__(self, args, nb_code=512, code_dim=512, output_emb_width=512, 
             down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
             activation='relu', norm=None):
    """
    Args:
        args: 配置参数，包含 dataname 属性
        nb_code (int): codebook 大小
        code_dim (int): 编码维度
        output_emb_width (int): 输出嵌入宽度
        down_t (int): 下采样次数
        stride_t (int): 下采样步长
        width (int): 网络宽度
        depth (int): 网络深度
        dilation_growth_rate (int): 膨胀增长率
        activation (str): 激活函数类型
        norm (str): 归一化类型
    """
```

#### 方法
```python
def encode(self, x):
    """
    编码动作序列为离散 tokens
    
    Args:
        x (torch.Tensor): 动作数据，形状 (batch, timesteps, features)
    
    Returns:
        torch.Tensor: 离散 tokens，形状 (batch, timesteps)
    """

def encode_x(self, x):
    """
    编码动作序列为连续特征
    
    Args:
        x (torch.Tensor): 动作数据，形状 (batch, timesteps, features)
    
    Returns:
        torch.Tensor: 连续特征，形状 (batch * timesteps, code_dim)
    """

def forward(self, x):
    """
    完整的前向传播（编码-量化-解码）
    
    Args:
        x (torch.Tensor): 输入动作数据
    
    Returns:
        tuple: (重构数据, 量化损失, 困惑度)
    """

def forward_decoder(self, x):
    """
    从 tokens 解码动作序列
    
    Args:
        x (torch.Tensor): 量化 tokens
    
    Returns:
        torch.Tensor: 解码的动作序列
    """
```

### 3.2 VQVAE_251

#### 类定义
```python
class VQVAE_251(nn.Module)
```

#### 初始化参数
```python
def __init__(self, args, nb_code=1024, code_dim=512, output_emb_width=512,
             down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
             activation='relu', norm=None):
    """
    Args:
        args: 配置参数，包含 dataname 和 quantizer 属性
        nb_code (int): codebook 大小
        code_dim (int): 编码维度
        output_emb_width (int): 输出嵌入宽度
        down_t (int): 下采样次数
        stride_t (int): 下采样步长
        width (int): 网络宽度
        depth (int): 网络深度
        dilation_growth_rate (int): 膨胀增长率
        activation (str): 激活函数类型
        norm (str): 归一化类型
    """
```

#### 方法
```python
def encode(self, x):
    """
    编码为离散 tokens
    
    Args:
        x (torch.Tensor): 输入数据
    
    Returns:
        torch.Tensor: 量化 tokens
    """

def encode_x(self, x):
    """
    编码为连续特征
    
    Args:
        x (torch.Tensor): 输入数据
    
    Returns:
        torch.Tensor: 编码特征
    """

def forward(self, x):
    """
    完整前向传播
    
    Args:
        x (torch.Tensor): 输入数据
    
    Returns:
        tuple: (重构数据, 量化损失, 困惑度)
    """

def forward_decoder(self, x):
    """
    从 tokens 解码
    
    Args:
        x (torch.Tensor): 量化 tokens
    
    Returns:
        torch.Tensor: 解码数据
    """
```

### 3.3 量化器类

#### QuantizeEMAReset

```python
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        """
        Args:
            nb_code (int): codebook 大小
            code_dim (int): 编码维度
            args: 配置参数，包含 mu (EMA 动量)
        """
    
    def quantize(self, x):
        """量化输入数据"""
    
    def dequantize(self, code_idx):
        """从索引解码"""
    
    def forward(self, x):
        """前向传播"""
```

#### QuantizeEMA

```python
class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        """初始化，mu 默认为 0.99"""
```

#### QuantizeReset

```python
class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        """初始化重置量化器"""
```

#### Quantizer

```python
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        """
        Args:
            n_e (int): 嵌入数量
            e_dim (int): 嵌入维度
            beta (float): commit loss 权重
        """
```

## 4. 网络组件 API

### 4.1 Encoder

#### 类定义
```python
class Encoder(nn.Module)
```

#### 初始化参数
```python
def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3,
             stride_t=2, width=512, depth=3, dilation_growth_rate=3,
             activation='relu', norm=None):
    """
    Args:
        input_emb_width (int): 输入嵌入宽度
        output_emb_width (int): 输出嵌入宽度
        down_t (int): 下采样次数
        stride_t (int): 下采样步长
        width (int): 网络宽度
        depth (int): 网络深度
        dilation_growth_rate (int): 膨胀增长率
        activation (str): 激活函数类型
        norm (str): 归一化类型
    """
```

### 4.2 Decoder

#### 类定义
```python
class Decoder(nn.Module)
```

#### 初始化参数
```python
def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3,
             stride_t=2, width=512, depth=3, dilation_growth_rate=3,
             activation='relu', norm=None):
    """
    参数与 Encoder 相同
    """
```

### 4.3 Resnet1D

#### 类定义
```python
class Resnet1D(nn.Module)
```

#### 初始化参数
```python
def __init__(self, n_in, n_depth, dilation_growth_rate=1,
             reverse_dilation=True, activation='relu', norm=None):
    """
    Args:
        n_in (int): 输入维度
        n_depth (int): ResNet 深度
        dilation_growth_rate (int): 膨胀增长率
        reverse_dilation (bool): 是否反向膨胀率
        activation (str): 激活函数类型
        norm (str): 归一化类型
    """
```

## 5. 运动编码器 API

### 5.1 MovementConvEncoder

#### 类定义
```python
class MovementConvEncoder(nn.Module)
```

#### 初始化参数
```python
def __init__(self, input_size, hidden_size, output_size):
    """
    Args:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层维度
        output_size (int): 输出维度
    """
```

#### 方法
```python
def forward(self, inputs):
    """
    Args:
        inputs (torch.Tensor): 输入张量，形状 (batch, seq_len, input_size)
    
    Returns:
        torch.Tensor: 编码特征，形状 (batch, seq_len, output_size)
    """
```

### 5.2 TextEncoderBiGRUCo

#### 类定义
```python
class TextEncoderBiGRUCo(nn.Module)
```

#### 初始化参数
```python
def __init__(self, word_size, pos_size, hidden_size, output_size, device):
    """
    Args:
        word_size (int): 词向量维度
        pos_size (int): 位置编码维度
        hidden_size (int): GRU 隐藏层维度
        output_size (int): 输出维度
        device (torch.device): 设备
    """
```

#### 方法
```python
def forward(self, word_embs, pos_onehot, cap_lens):
    """
    Args:
        word_embs (torch.Tensor): 词嵌入
        pos_onehot (torch.Tensor): 位置 one-hot 编码
        cap_lens (torch.Tensor): 序列长度
    
    Returns:
        torch.Tensor: 文本编码
    """
```

### 5.3 MotionEncoderBiGRUCo

#### 类定义
```python
class MotionEncoderBiGRUCo(nn.Module)
```

#### 初始化参数
```python
def __init__(self, input_size, hidden_size, output_size, device):
    """
    Args:
        input_size (int): 输入维度
        hidden_size (int): GRU 隐藏层维度
        output_size (int): 输出维度
        device (torch.device): 设备
    """
```

#### 方法
```python
def forward(self, inputs, m_lens):
    """
    Args:
        inputs (torch.Tensor): 运动数据
        m_lens (torch.Tensor): 运动序列长度
    
    Returns:
        torch.Tensor: 运动编码
    """
```

## 6. 评估包装器 API

### 6.1 EvaluatorModelWrapper

#### 类定义
```python
class EvaluatorModelWrapper(object)
```

#### 初始化参数
```python
def __init__(self, opt):
    """
    Args:
        opt: 配置对象，必须包含:
            - dataset_name (str): 数据集名称
            - checkpoints_dir (str): 检查点目录
            - device (torch.device): 设备
    """
```

#### 方法
```python
def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
    """
    获取文本和动作的联合嵌入
    
    Args:
        word_embs (torch.Tensor): 词嵌入
        pos_ohot (torch.Tensor): 位置编码
        cap_lens (torch.Tensor): 文本长度
        motions (torch.Tensor): 运动数据
        m_lens (torch.Tensor): 运动长度
    
    Returns:
        tuple: (文本嵌入, 运动嵌入)
    """

def get_motion_embeddings(self, motions, m_lens):
    """
    获取运动嵌入
    
    Args:
        motions (torch.Tensor): 运动数据
        m_lens (torch.Tensor): 运动长度
    
    Returns:
        torch.Tensor: 运动嵌入
    """
```

### 6.2 build_models

#### 函数签名
```python
def build_models(opt):
    """
    构建评估模型
    
    Args:
        opt: 配置对象
    
    Returns:
        tuple: (文本编码器, 运动编码器, 运动卷积编码器)
    """
```

## 7. 常量定义

### 7.1 模态常量

#### X_TOKEN_INDEX
```python
X_TOKEN_INDEX = {
    'IMAGE': -200,
    'VIDEO': -201,
    'AUDIO': -202,
    'THERMAL': -203,
    'DEPTH': -204
}
```

#### DEFAULT_X_TOKEN
```python
DEFAULT_X_TOKEN = {
    'IMAGE': "<image>",
    'VIDEO': "<video>",
    'AUDIO': "<audio>",
    'THERMAL': "<thermal>",
    'DEPTH': "<depth>"
}
```

#### DEFAULT_X_PATCH_TOKEN
```python
DEFAULT_X_PATCH_TOKEN = {
    'IMAGE': "<im_patch>",
    'VIDEO': "<vi_patch>",
    'AUDIO': "<au_patch>",
    'THERMAL': "<th_patch>",
    'DEPTH': "<de_patch>"
}
```

## 8. 使用示例

### 8.1 构建完整模型

```python
from models.multimodal_encoder.builder import build_image_tower, build_video_tower
from models.multimodal_projector.builder import build_vision_projector
from models.vqvae import HumanVQVAE

# 构建编码器
image_encoder = build_image_tower(config)
video_encoder = build_video_tower(config)

# 构建投影器
vision_projector = build_vision_projector(config)

# 构建动作编码器
motion_encoder = HumanVQVAE(args)

# 前向传播
image_features = image_encoder(images)
video_features = video_encoder(videos)

# 特征投影
projected_image = vision_projector(image_features)
projected_video = vision_projector(video_features)

# 动作编码
motion_tokens = motion_encoder.encode(motions)
```

### 8.2 评估使用

```python
from models.evaluator_wrapper import EvaluatorModelWrapper

# 初始化评估器
evaluator = EvaluatorModelWrapper(opt)

# 获取联合嵌入
text_embedding, motion_embedding = evaluator.get_co_embeddings(
    word_embs, pos_ohot, cap_lens, motions, m_lens
)

# 计算相似度
similarity = torch.cosine_similarity(text_embedding, motion_embedding)
```

这个 API 文档提供了 MotionLLM 各模块的完整接口说明，便于开发者理解和使用各个组件。