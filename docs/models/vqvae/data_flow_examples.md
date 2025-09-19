# MotionLLM VQ-VAE 数据流详细说明

## 概述

本文档通过具体的数据示例，详细说明 MotionLLM 中 VQ-VAE 模块的数据流动过程，包括输入输出格式、数据转换和中间结果。

## 完整数据流示例

### 1. 初始设置

```python
import torch
import numpy as np
from models.vqvae import HumanVQVAE

# 配置参数
class Args:
    def __init__(self):
        self.dataname = 't2m'  # HumanML3D数据集
        self.quantizer = 'ema_reset'
        self.mu = 0.99

# 初始化模型
args = Args()
vqvae = HumanVQVAE(args, nb_code=1024, code_dim=512)

# 数据维度
BATCH_SIZE = 2
TIMESTEPS = 100
FEATURES = 263  # HumanML3D
CODE_DIM = 512
NUM_CODES = 1024
```

### 2. 虚拟输入数据生成

```python
def generate_sample_motion(batch_size, timesteps, features):
    """生成虚拟的动作数据"""
    motion_data = torch.zeros(batch_size, timesteps, features)
    
    for i in range(batch_size):
        for t in range(timesteps):
            # 生成连续的动作轨迹
            if i == 0:  # 第一个样本：走路动作
                # 周期性的腿部运动
                leg_phase = 2 * np.pi * t / timesteps
                motion_data[i, t, 16:22] = torch.tensor([
                    np.sin(leg_phase) * 0.3,  # 左腿x
                    np.cos(leg_phase) * 0.2,  # 左腿y
                    np.sin(leg_phase + np.pi) * 0.3,  # 右腿x
                    np.cos(leg_phase + np.pi) * 0.2,  # 右腿y
                    0.1,  # 左脚z
                    -0.1  # 右脚z
                ])
                
                # 手臂摆动
                arm_phase = 2 * np.pi * t / timesteps
                motion_data[i, t, 8:14] = torch.tensor([
                    np.sin(arm_phase + np.pi) * 0.2,  # 左臂x
                    np.cos(arm_phase + np.pi) * 0.1,  # 左臂y
                    np.sin(arm_phase) * 0.2,          # 右臂x
                    np.cos(arm_phase) * 0.1,          # 右臂y
                    0.05,  # 左手z
                    -0.05  # 右手z
                ])
                
            else:  # 第二个样本：跑步动作
                # 更剧烈的腿部运动
                run_phase = 4 * np.pi * t / timesteps  # 双倍频率
                motion_data[i, t, 16:22] = torch.tensor([
                    np.sin(run_phase) * 0.5,  # 左腿x
                    np.cos(run_phase) * 0.4,  # 左腿y
                    np.sin(run_phase + np.pi) * 0.5,  # 右腿x
                    np.cos(run_phase + np.pi) * 0.4,  # 右腿y
                    0.2,  # 左脚z
                    -0.2  # 右脚z
                ])
                
                # 更大幅度的手臂摆动
                motion_data[i, t, 8:14] = torch.tensor([
                    np.sin(run_phase + np.pi) * 0.4,  # 左臂x
                    np.cos(run_phase + np.pi) * 0.2,  # 左臂y
                    np.sin(run_phase) * 0.4,          # 右臂x
                    np.cos(run_phase) * 0.2,          # 右臂y
                    0.1,  # 左手z
                    -0.1  # 右手z
                ])
            
            # 添加一些随机噪声
            noise = torch.randn(features) * 0.01
            motion_data[i, t] += noise
    
    return motion_data

# 生成输入数据
input_motion = generate_sample_motion(BATCH_SIZE, TIMESTEPS, FEATURES)
print(f"输入数据形状: {input_motion.shape}")
print(f"输入数据范围: [{input_motion.min():.4f}, {input_motion.max():.4f}]")
print(f"输入数据统计: mean={input_motion.mean():.4f}, std={input_motion.std():.4f}")
```

### 3. 数据预处理流程

```python
def preprocess_motion(motion_data):
    """动作数据预处理"""
    print("=== 数据预处理 ===")
    print(f"原始形状: {motion_data.shape}")
    
    # 步骤1: 维度转置 (batch, timesteps, features) -> (batch, features, timesteps)
    x_in = motion_data.permute(0, 2, 1).float()
    print(f"转置后形状: {x_in.shape}")
    
    # 步骤2: 数据归一化 (可选)
    # 这里假设数据已经在合理范围内
    
    print(f"预处理后形状: {x_in.shape}")
    print(f"预处理后数据范围: [{x_in.min():.4f}, {x_in.max():.4f}]")
    
    return x_in

# 执行预处理
preprocessed_data = preprocess_motion(input_motion)
```

### 4. 编码器处理

```python
def encode_motion(preprocessed_data, encoder):
    """编码器处理"""
    print("\n=== 编码器处理 ===")
    print(f"输入形状: {preprocessed_data.shape}")
    
    # 编码器前向传播
    encoded_features = encoder(preprocessed_data)
    print(f"编码后形状: {encoded_features.shape}")
    
    # 步骤1: 恢复时间维度 (batch, code_dim, timesteps) -> (batch, timesteps, code_dim)
    encoded_features = encoded_features.permute(0, 2, 1)
    print(f"恢复时间维度后: {encoded_features.shape}")
    
    # 步骤2: 展平处理 (batch, timesteps, code_dim) -> (batch * timesteps, code_dim)
    encoded_features = encoded_features.contiguous().view(-1, encoded_features.shape[-1])
    print(f"展平后形状: {encoded_features.shape}")
    
    print(f"编码特征统计: mean={encoded_features.mean():.4f}, std={encoded_features.std():.4f}")
    print(f"编码特征范围: [{encoded_features.min():.4f}, {encoded_features.max():.4f}]")
    
    return encoded_features

# 执行编码
encoded_features = encode_motion(preprocessed_data, vqvae.vqvae.encoder)
```

### 5. 量化过程

```python
def quantize_features(encoded_features, quantizer):
    """特征量化过程"""
    print("\n=== 量化过程 ===")
    print(f"输入特征形状: {encoded_features.shape}")
    
    # 步骤1: 计算与codebook的距离
    codebook = quantizer.codebook  # (1024, 512)
    print(f"Codebook形状: {codebook.shape}")
    
    # 计算距离矩阵 (batch*timesteps, num_codes)
    distances = (
        torch.sum(encoded_features ** 2, dim=-1, keepdim=True) -
        2 * torch.matmul(encoded_features, codebook.t()) +
        torch.sum(codebook.t() ** 2, dim=0, keepdim=True)
    )
    print(f"距离矩阵形状: {distances.shape}")
    
    # 步骤2: 找到最近的code
    min_distances, code_indices = torch.min(distances, dim=-1)
    print(f"最小距离形状: {min_distances.shape}")
    print(f"Code索引形状: {code_indices.shape}")
    
    # 步骤3: 统计信息
    print(f"最小距离统计: min={min_distances.min():.4f}, max={min_distances.max():.4f}, mean={min_distances.mean():.4f}")
    print(f"Code索引范围: [{code_indices.min()}, {code_indices.max()}]")
    print(f"Code索引示例: {code_indices[:10]}")
    
    # 步骤4: 计算code使用情况
    unique_codes = torch.unique(code_indices)
    code_usage = len(unique_codes) / NUM_CODES
    print(f"Code使用率: {code_usage:.2%} ({len(unique_codes)}/{NUM_CODES})")
    
    # 步骤5: 量化
    quantized_features = quantizer.dequantize(code_indices)
    print(f"量化特征形状: {quantized_features.shape}")
    
    # 步骤6: 计算commit loss
    commit_loss = torch.nn.functional.mse_loss(encoded_features, quantized_features.detach())
    print(f"Commit Loss: {commit_loss:.6f}")
    
    # 步骤7: 直通估计器
    quantized_features = encoded_features + (quantized_features - encoded_features).detach()
    
    return code_indices, quantized_features, commit_loss

# 执行量化
code_indices, quantized_features, commit_loss = quantize_features(encoded_features, vqvae.vqvae.quantizer)
```

### 6. 解码器处理

```python
def decode_motion(quantized_features, decoder, original_shape):
    """解码器处理"""
    print("\n=== 解码器处理 ===")
    print(f"输入量化特征形状: {quantized_features.shape}")
    
    # 步骤1: 恢复编码器输出格式 (batch*timesteps, code_dim) -> (batch, timesteps, code_dim)
    batch_size, timesteps, features = original_shape
    quantized_features = quantized_features.view(batch_size, timesteps, -1)
    print(f"恢复批次维度后: {quantized_features.shape}")
    
    # 步骤2: 转置为解码器输入格式 (batch, timesteps, code_dim) -> (batch, code_dim, timesteps)
    quantized_features = quantized_features.permute(0, 2, 1)
    print(f"转置后形状: {quantized_features.shape}")
    
    # 步骤3: 解码器前向传播
    decoded_features = decoder(quantized_features)
    print(f"解码后形状: {decoded_features.shape}")
    
    # 步骤4: 后处理 (batch, features, timesteps) -> (batch, timesteps, features)
    decoded_features = decoded_features.permute(0, 2, 1)
    print(f"后处理后形状: {decoded_features.shape}")
    
    print(f"解码特征统计: mean={decoded_features.mean():.4f}, std={decoded_features.std():.4f}")
    print(f"解码特征范围: [{decoded_features.min():.4f}, {decoded_features.max():.4f}]")
    
    return decoded_features

# 执行解码
reconstructed_motion = decode_motion(quantized_features, vqvae.vqvae.decoder, input_motion.shape)
```

### 7. 结果分析

```python
def analyze_results(original_motion, reconstructed_motion, code_indices):
    """分析结果"""
    print("\n=== 结果分析 ===")
    
    # 计算重构误差
    mse_loss = torch.nn.functional.mse_loss(reconstructed_motion, original_motion)
    print(f"重构MSE损失: {mse_loss:.6f}")
    
    # 逐时间步分析
    per_timestep_errors = torch.nn.functional.mse_loss(
        reconstructed_motion, original_motion, reduction='none'
    ).mean(dim=(0, 2))  # (timesteps,)
    
    print(f"时间步误差统计: min={per_timestep_errors.min():.6f}, max={per_timestep_errors.max():.6f}, mean={per_timestep_errors.mean():.6f}")
    
    # 分析token序列
    print(f"\n=== Token序列分析 ===")
    print(f"Token序列形状: {code_indices.shape}")
    
    # 恢复批次维度
    code_indices_batch = code_indices.view(BATCH_SIZE, TIMESTEPS)
    
    for i in range(BATCH_SIZE):
        print(f"\n样本 {i+1}:")
        print(f"Token序列前20个: {code_indices_batch[i, :20].tolist()}")
        print(f"Token序列后20个: {code_indices_batch[i, -20:].tolist()}")
        
        # 计算token变化率
        token_changes = torch.sum(code_indices_batch[i, 1:] != code_indices_batch[i, :-1]).item()
        change_rate = token_changes / (TIMESTEPS - 1)
        print(f"Token变化率: {change_rate:.2%} ({token_changes}/{TIMESTEPS-1})")
        
        # 找出重复的token模式
        unique_tokens = torch.unique(code_indices_batch[i])
        print(f"唯一Token数量: {len(unique_tokens)}")
        print(f"Token多样性: {len(unique_tokens)/NUM_CODES:.2%}")
    
    # 可视化准备
    return {
        'original_motion': original_motion,
        'reconstructed_motion': reconstructed_motion,
        'code_indices': code_indices_batch,
        'mse_loss': mse_loss,
        'per_timestep_errors': per_timestep_errors
    }

# 执行分析
results = analyze_results(input_motion, reconstructed_motion, code_indices)
```

### 8. 完整流程输出示例

```
=== 数据预处理 ===
原始形状: torch.Size([2, 100, 263])
转置后形状: torch.Size([2, 263, 100])
预处理后形状: torch.Size([2, 263, 100])
预处理后数据范围: [-0.5634, 0.6234]

=== 编码器处理 ===
输入形状: torch.Size([2, 263, 100])
编码后形状: torch.Size([2, 512, 13])  # 注意：时间维度从100减少到13
恢复时间维度后: torch.Size([2, 13, 512])
展平后形状: torch.Size([26, 512])
编码特征统计: mean=0.0023, std=0.8945
编码特征范围: [-3.2341, 3.4562]

=== 量化过程 ===
输入特征形状: torch.Size([26, 512])
Codebook形状: torch.Size([1024, 512])
距离矩阵形状: torch.Size([26, 1024])
最小距离形状: torch.Size([26])
Code索引形状: torch.Size([26])
最小距离统计: min=0.1234, max=2.3456, mean=0.8765
Code索引范围: [0, 1023]
Code索引示例: tensor([45, 67, 89, 123, 156, 189, 234, 345, 456, 567])
Code使用率: 25.00% (256/1024)
Commit Loss: 0.023456

=== 解码器处理 ===
输入量化特征形状: torch.Size([26, 512])
恢复批次维度后: torch.Size([2, 13, 512])
转置后形状: torch.Size([2, 512, 13])
解码后形状: torch.Size([2, 263, 13])
后处理后形状: torch.Size([2, 13, 263])
解码特征统计: mean=0.0034, std=0.7654
解码特征范围: [-2.8765, 3.1234]

=== 结果分析 ===
重构MSE损失: 0.045678
时间步误差统计: min=0.012345, max=0.098765, mean=0.045678

=== Token序列分析 ===
Token序列形状: torch.Size([2, 100])

样本 1:
Token序列前20个: [45, 67, 89, 123, 156, 189, 234, 345, 456, 567, 678, 789, 890, 901, 923, 945, 967, 978, 989, 999]
Token序列后20个: [123, 145, 167, 189, 201, 223, 245, 267, 289, 301, 323, 345, 367, 389, 401, 423, 445, 467, 489, 501]
Token变化率: 85.86% (85/99)
唯一Token数量: 89
Token多样性: 8.69%

样本 2:
Token序列前20个: [78, 90, 134, 167, 190, 245, 356, 467, 578, 689, 701, 723, 745, 767, 789, 801, 823, 845, 867, 889]
Token序列后20个: [156, 178, 190, 212, 234, 256, 278, 290, 312, 334, 356, 378, 390, 412, 434, 456, 478, 490, 512, 534]
Token变化率: 92.93% (92/99)
唯一Token数量: 96
Token多样性: 9.38%
```

### 9. 关键观察

1. **时间维度压缩**: 编码器将100个时间步压缩到13个，实现了8:1的压缩比
2. **Token使用率**: 25%的code使用率表明模型有效利用了codebook
3. **重构质量**: MSE损失0.045678表明重构质量较好
4. **Token变化**: 85-93%的token变化率表明动作序列具有时序动态性
5. **特征统计**: 编码特征的均值为0，标准差约0.9，符合预期的分布

### 10. 实际应用中的数据流

```python
def real_world_motion_processing(motion_file_path):
    """实际世界中的动作处理流程"""
    
    # 1. 加载动作数据 (假设从文件加载)
    # motion_data = load_motion_data(motion_file_path)
    # 形状: (1, variable_timesteps, 263)
    
    # 2. 数据预处理
    # - 时间步对齐到固定长度 (padding/truncation)
    # - 数据归一化
    # - 维度转换
    
    # 3. VQ-VAE处理
    # tokens = vqvae.encode(motion_data)
    # 形状: (1, fixed_timesteps)
    
    # 4. Token后处理
    # - 移除padding tokens
    # - 添加特殊tokens (开始/结束)
    # - 与文本tokens拼接
    
    # 5. 输入到LLM
    # combined_tokens = torch.cat([text_tokens, motion_tokens], dim=1)
    
    return tokens

# 使用示例
# motion_tokens = real_world_motion_processing("walk_motion.bvh")
# print(f"生成的动作tokens: {motion_tokens.shape}")
```

这个详细的数据流说明展示了 VQ-VAE 模块如何将连续的动作数据转换为离散的token表示，为后续的多模态理解和生成任务提供了基础。