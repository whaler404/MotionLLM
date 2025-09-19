# MotionLLM 模型目录

本文档提供了 MotionLLM 模型模块的完整文档导航。

## 文档结构

### 1. 架构概览
- **[架构概览](./architecture_overview.md)**
  - 系统整体架构
  - 核心模块介绍
  - 设计理念和技术选型

### 2. 模块耦合关系
- **[模块耦合关系](./module_coupling.md)**
  - 模块间依赖关系图
  - 数据流向分析
  - 接口设计和解耦策略

### 3. 实现细节
- **[实现细节](./implementation_details.md)**
  - 各模块的代码实现
  - 关键算法和数据处理流程
  - 性能优化策略

### 4. API 参考
- **[API 参考](./api_reference.md)**
  - 完整的 API 接口文档
  - 方法签名和参数说明
  - 使用示例

### 5. 配置指南
- **[配置指南](./configuration_guide.md)**
  - 详细的配置参数说明
  - 推荐配置组合
  - 配置验证和优化

## 快速导航

### 新用户开始
1. 先阅读 [架构概览](./architecture_overview.md) 了解系统整体设计
2. 查看 [配置指南](./configuration_guide.md) 了解如何配置模型
3. 参考 [API 参考](./api_reference.md) 了解具体使用方法

### 深入理解
1. 阅读 [模块耦合关系](./module_coupling.md) 理解模块间交互
2. 学习 [实现细节](./implementation_details.md) 了解具体实现
3. 根据需要调整 [配置指南](./configuration_guide.md) 中的参数

### 开发者参考
- 模块文件位置：`/models/`
- 多模态编码器：`/models/multimodal_encoder/`
- 多模态投影器：`/models/multimodal_projector/`
- VQ-VAE 模块：`/models/vqvae.py`
- 网络组件：`/models/encdec.py`, `/models/resnet.py`
- 运动编码器：`/models/modules.py`
- 评估包装器：`/models/evaluator_wrapper.py`

## 相关文档

- 项目主文档：`/docs/README.md`
- 配置文档：`/docs/configuration.md`
- 数据处理：`/docs/data_processing.md`
- 训练推理：`/docs/training_inference.md`

## 常见问题

### Q: 如何选择合适的投影器类型？
A: 参考 [配置指南](./configuration_guide.md#22-支持的投影器类型) 部分，推荐使用 `mlp2x_gelu` 作为默认选择。

### Q: 如何配置 VQ-VAE 参数？
A: 查看 [配置指南](./configuration_guide.md#3-vq-vae-配置) 部分，包含详细的参数说明和推荐配置。

### Q: 如何扩展新的模态？
A: 参考 [模块耦合关系](./module_coupling.md#扩展点设计) 部分，了解扩展点设计。

### Q: 如何进行模型评估？
A: 查看 [API 参考](./api_reference.md#6-评估包装器-api) 部分，了解评估接口的使用方法。

## 版本信息

- 文档版本：1.0
- 对应代码版本：main 分支
- 最后更新：2024年

## 贡献指南

如果您发现文档中的错误或有改进建议，请：
1. 检查对应源代码的实现
2. 确认配置参数的正确性
3. 提交 Issue 或 Pull Request

## 技术支持

- 代码问题：查看源代码实现
- 配置问题：参考配置指南
- 架构问题：阅读架构概览
- API 问题：查阅 API 参考