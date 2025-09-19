# MotionLLM 多模态编码器概览

## 概述

MotionLLM 支持多种多模态编码器，包括 CLIP、LanguageBind 和 MAE 等不同类型的编码器。这些编码器负责将图像、视频等多模态数据转换为特征向量，为后续的多模态理解和生成任务提供基础。

## 支持的编码器类型

### 1. CLIP 编码器
- **用途**: 图像编码
- **模型**: OpenAI CLIP 系列
- **特点**: 简单高效，支持多种预训练模型

### 2. LanguageBind 编码器
- **用途**: 图像、视频、音频、热成像、深度图像
- **特点**: 统一的多模态框架，支持跨模态对齐

### 3. MAE 编码器
- **用途**: 图像编码
- **特点**: 掩码自编码器，具有较强的表示学习能力

## 编码器选择策略

### Builder 模式
```python
def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    
    # CLIP 编码器
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    
    # LanguageBind 图像编码器
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    
    # MAE 编码器
    if 'mae' in image_tower:
        return MAEVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    
    raise ValueError(f'Unknown image tower: {image_tower}')
```

### 视频编码器
```python
def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    
    # 目前只支持 LanguageBind 视频编码器
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    
    raise ValueError(f'Unknown video tower: {video_tower}')
```

## 数据流概述

### 输入数据
- **图像**: `(batch, channels, height, width)`
- **视频**: `(batch, channels, frames, height, width)`

### 输出数据
- **图像特征**: `(batch, num_patches, hidden_size)`
- **视频特征**: `(batch, frames * num_patches, hidden_size)`

### 处理流程
```
原始数据 → 预处理 → 编码器 → 特征选择 → 输出特征
```

## 配置参数

### 通用参数
- `mm_vision_select_layer`: 选择隐藏层的索引 (默认: -2)
- `mm_vision_select_feature`: 特征类型 ('patch' 或 'cls_patch')

### 编码器特定参数
- **CLIP**: `image_size`, `patch_size`, `hidden_size`
- **LanguageBind**: `num_frames`, `temporal_embedding`
- **MAE**: `mask_ratio`, `decoder_depth`

## 使用示例

### 基本使用
```python
from models.multimodal_encoder.builder import build_image_tower, build_video_tower

# 构建图像编码器
image_encoder = build_image_tower(config)

# 构建视频编码器
video_encoder = build_video_tower(config)

# 编码图像
image_features = image_encoder(images)

# 编码视频
video_features = video_encoder(videos)
```

### 延迟加载
```python
# 延迟加载模式
image_encoder = build_image_tower(config, delay_load=True)

# 需要时手动加载
image_encoder.load_model()
```

## 详细文档

各编码器的详细说明请参考对应的文档：

- [CLIP 编码器详细说明](./clip_encoder_guide.md)
- [LanguageBind 编码器详细说明](./languagebind_encoder_guide.md)
- [MAE 编码器详细说明](./mae_encoder_guide.md)

## 性能比较

| 编码器类型 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| CLIP | 简单快速，模型丰富 | 单一模态 | 基础图像理解 |
| LanguageBind | 多模态统一，跨模态对齐 | 计算复杂度高 | 复杂多模态任务 |
| MAE | 表示能力强，自监督训练 | 推理速度慢 | 高质量特征提取 |

## 最佳实践

### 编码器选择
- **基础任务**: 使用 CLIP 编码器
- **多模态任务**: 使用 LanguageBind 编码器
- **高质量特征**: 使用 MAE 编码器

### 性能优化
- 使用延迟加载减少内存占用
- 冻结编码器参数，只训练投影器
- 选择合适的特征层

## 故障排除

### 常见问题
1. **内存不足**: 使用延迟加载或减小批次大小
2. **模型加载失败**: 检查网络连接和模型路径
3. **特征维度不匹配**: 确认编码器和投影器配置

### 调试工具
```python
# 检查编码器状态
print(f"编码器已加载: {encoder.is_loaded}")
print(f"隐藏层大小: {encoder.hidden_size}")
print(f"特征维度: {encoder.num_patches}")
```

这个概览文档提供了 MotionLLM 多模态编码器的整体介绍，详细的编码器说明请参考各编码器的专门文档。