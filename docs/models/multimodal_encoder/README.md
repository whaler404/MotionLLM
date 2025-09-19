# MotionLLM 多模态编码器文档

## 概述

本目录包含 MotionLLM 项目中多模态编码器的详细文档。多模态编码器负责将图像、视频等多模态数据转换为特征向量，为后续的多模态理解和生成任务提供基础。

## 文档结构

### 核心文档

- **[概览文档](./overview.md)** - 多模态编码器的整体介绍，包括编码器类型、选择策略和数据流
- **[CLIP 编码器](./clip_encoder_guide.md)** - OpenAI CLIP 图像编码器的详细说明和使用指南
- **[LanguageBind 编码器](./languagebind_encoder_guide.md)** - LanguageBind 多模态编码器的详细说明，支持图像、视频等多种模态
- **[MAE 编码器](./mae_encoder_guide.md)** - 掩码自编码器（MAE）的详细说明，基于自监督学习

### 快速导航

#### 如果您需要：
- **了解所有编码器类型** → 阅读 [概览文档](./overview.md)
- **使用基础的图像编码** → 阅读 [CLIP 编码器](./clip_encoder_guide.md)
- **处理视频或多模态数据** → 阅读 [LanguageBind 编码器](./languagebind_encoder_guide.md)
- **需要高质量特征表示** → 阅读 [MAE 编码器](./mae_encoder_guide.md)

## 编码器对比

| 编码器类型 | 支持模态 | 特点 | 适用场景 |
|-----------|---------|------|---------|
| **CLIP** | 图像 | 简单高效，模型丰富 | 基础图像理解 |
| **LanguageBind** | 图像、视频、音频、热成像、深度 | 多模态统一，跨模态对齐 | 复杂多模态任务 |
| **MAE** | 图像 | 自监督学习，表示能力强 | 高质量特征提取 |

## 数据格式总览

### 图像数据
- **输入格式**: `(batch, channels, height, width)`
- **标准尺寸**: `(batch, 3, 224, 224)`
- **输出格式**: `(batch, num_patches, hidden_size)`

### 视频数据
- **输入格式**: `(batch, channels, frames, height, width)`
- **标准尺寸**: `(batch, 3, 8, 224, 224)`
- **输出格式**: `(batch, frames * num_patches, hidden_size)`

## 使用示例

### 基本使用
```python
from models.multimodal_encoder.builder import build_image_tower, build_video_tower

# 构建编码器
image_encoder = build_image_tower(config)
video_encoder = build_video_tower(config)

# 编码数据
image_features = image_encoder(images)
video_features = video_encoder(videos)
```

### 延迟加载
```python
# 延迟加载模式
image_encoder = build_image_tower(config, delay_load=True)

# 需要时手动加载
image_encoder.load_model()
```

## 编码器选择指南

### 性能优先
- **CLIP**: 推理速度快，内存占用小
- **LanguageBind**: 功能强大，但计算复杂度高
- **MAE**: 特征质量高，但推理速度较慢

### 功能需求
- **纯图像处理**: CLIP 或 MAE
- **视频处理**: LanguageBind
- **多模态融合**: LanguageBind
- **自监督学习**: MAE

## 配置参数

### 通用参数
- `mm_vision_select_layer`: 选择隐藏层的索引 (默认: -2)
- `mm_vision_select_feature`: 特征类型 ('patch' 或 'cls_patch')

### 编码器特定参数
- **CLIP**: `image_size`, `patch_size`, `hidden_size`
- **LanguageBind**: `num_frames`, `temporal_embedding`
- **MAE**: `mask_ratio`, `decoder_depth`

## 最佳实践

1. **编码器选择**: 根据具体任务需求选择合适的编码器
2. **性能优化**: 使用延迟加载减少内存占用
3. **特征选择**: 选择合适的特征层和特征类型
4. **批处理**: 合理设置批次大小以平衡性能和内存使用

## 调试工具

```python
# 检查编码器状态
print(f"编码器已加载: {encoder.is_loaded}")
print(f"隐藏层大小: {encoder.hidden_size}")
print(f"特征维度: {encoder.num_patches}")
```

## 相关资源

- [MotionLLM 主文档](../README.md)
- [模型架构概览](../architecture_overview.md)
- [模块耦合关系](../module_coupling.md)
- [实现细节](../implementation_details.md)

## 更新日志

- **2024-01-18**: 完成所有编码器的详细文档
- **2024-01-17**: 创建多模态编码器文档结构
- **2024-01-16**: 开始多模态编码器文档编写

## 贡献

如果您发现文档中的错误或有改进建议，请提交 Issue 或 Pull Request。

## 许可证

本文档遵循 MotionLLM 项目的许可证条款。