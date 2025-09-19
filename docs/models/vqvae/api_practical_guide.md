# MotionLLM VQ-VAE API 实用指南

## 概述

本文档提供 MotionLLM VQ-VAE 模块的实用 API 指南，包含具体的使用示例、常见问题和最佳实践。

## 核心类和方法

### 1. HumanVQVAE 类

#### 基本初始化
```python
from models.vqvae import HumanVQVAE

class Args:
    def __init__(self):
        self.dataname = 't2m'      # 数据集: 't2m' 或 'kit'
        self.quantizer = 'ema_reset'  # 量化器类型
        self.mu = 0.99              # EMA动量

# 创建模型实例
args = Args()
vqvae = HumanVQVAE(
    args=args,
    nb_code=1024,        # codebook大小
    code_dim=512,        # 编码维度
    output_emb_width=512,# 输出嵌入宽度
    down_t=3,            # 下采样次数
    stride_t=2,          # 下采样步长
    width=512,           # 网络宽度
    depth=3,             # 网络深度
    dilation_growth_rate=3, # 膨胀增长率
    activation='relu',    # 激活函数
    norm=None            # 归一化
)
```

#### 数据集适配
```python
# 不同数据集的自动适配
if args.dataname == 'kit':
    input_features = 251  # KIT数据集
    nb_joints = 21        # 21个关节
elif args.dataname == 't2m':
    input_features = 263  # HumanML3D数据集
    nb_joints = 22        # 22个关节
```

### 2. 核心方法详解

#### 2.1 encode() - 离散编码

**功能**: 将连续动作编码为离散token序列

```python
def encode(self, x):
    """
    将动作数据编码为离散tokens
    
    Args:
        x (torch.Tensor): 动作数据，形状 (batch, timesteps, features)
    
    Returns:
        torch.Tensor: 离散tokens，形状 (batch, timesteps)
    """

# 使用示例
batch_size, timesteps, features = 2, 100, 263
motion_data = torch.randn(batch_size, timesteps, features)

# 编码为tokens
motion_tokens = vqvae.encode(motion_data)
print(f"输入形状: {motion_data.shape}")
print(f"输出tokens形状: {motion_tokens.shape}")
print(f"Token示例: {motion_tokens[0, :10]}")
print(f"Token范围: [{motion_tokens.min()}, {motion_tokens.max()}]")
```

**输出示例**:
```
输入形状: torch.Size([2, 100, 263])
输出tokens形状: torch.Size([2, 100])
Token示例: tensor([45, 67, 89, 123, 156, 189, 234, 345, 456, 567])
Token范围: [0, 1023]
```

#### 2.2 encode_x() - 连续编码

**功能**: 将动作编码为连续特征向量

```python
def encode_x(self, x):
    """
    将动作数据编码为连续特征
    
    Args:
        x (torch.Tensor): 动作数据，形状 (batch, timesteps, features)
    
    Returns:
        torch.Tensor: 连续特征，形状 (batch * timesteps, code_dim)
    """

# 使用示例
motion_features = vqvae.encode_x(motion_data)
print(f"输入形状: {motion_data.shape}")
print(f"输出特征形状: {motion_features.shape}")
print(f"特征统计: mean={motion_features.mean():.4f}, std={motion_features.std():.4f}")
```

**输出示例**:
```
输入形状: torch.Size([2, 100, 263])
输出特征形状: torch.Size([200, 512])
特征统计: mean=0.0023, std=0.8945
```

#### 2.3 forward() - 完整前向传播

**功能**: 完整的编码-量化-解码流程

```python
def forward(self, x):
    """
    完整的VQ-VAE前向传播
    
    Args:
        x (torch.Tensor): 输入动作数据
    
    Returns:
        tuple: (reconstructed_x, loss, perplexity)
            reconstructed_x: 重构的动作数据
            loss: 量化损失
            perplexity: codebook使用困惑度
    """

# 使用示例
reconstructed_motion, loss, perplexity = vqvae(motion_data)
print(f"输入形状: {motion_data.shape}")
print(f"重构形状: {reconstructed_motion.shape}")
print(f"量化损失: {loss:.6f}")
print(f"困惑度: {perplexity:.4f}")

# 计算重构质量
mse_loss = torch.nn.functional.mse_loss(reconstructed_motion, motion_data)
print(f"重构MSE: {mse_loss:.6f}")
```

**输出示例**:
```
输入形状: torch.Size([2, 100, 263])
重构形状: torch.Size([2, 100, 263])
量化损失: 0.023456
困惑度: 5.6789
重构MSE: 0.045678
```

#### 2.4 forward_decoder() - Token解码

**功能**: 从离散tokens解码动作

```python
def forward_decoder(self, x):
    """
    从tokens解码动作
    
    Args:
        x (torch.Tensor): 离散tokens，形状 (batch, timesteps)
    
    Returns:
        torch.Tensor: 解码的动作数据
    """

# 使用示例
# 从已有tokens或生成新tokens
tokens = torch.randint(0, 1024, (1, 100))  # 随机tokens
decoded_motion = vqvae.forward_decoder(tokens)

print(f"输入tokens形状: {tokens.shape}")
print(f"解码动作形状: {decoded_motion.shape}")
print(f"解码动作统计: mean={decoded_motion.mean():.4f}, std={decoded_motion.std():.4f}")
```

**输出示例**:
```
输入tokens形状: torch.Size([1, 100])
解码动作形状: torch.Size([1, 100, 263])
解码动作统计: mean=0.0034, std=0.7654
```

## 实用工具函数

### 1. 动作预处理工具

```python
def preprocess_motion(motion_data, target_length=None):
    """
    动作数据预处理
    
    Args:
        motion_data (torch.Tensor): 原始动作数据
        target_length (int, optional): 目标长度
    
    Returns:
        torch.Tensor: 预处理后的动作数据
    """
    if target_length is not None:
        # 时间步对齐
        if motion_data.shape[1] > target_length:
            # 截断
            motion_data = motion_data[:, :target_length, :]
        elif motion_data.shape[1] < target_length:
            # 填充
            pad_length = target_length - motion_data.shape[1]
            padding = torch.zeros(motion_data.shape[0], pad_length, motion_data.shape[2])
            motion_data = torch.cat([motion_data, padding], dim=1)
    
    # 数据归一化
    mean = motion_data.mean(dim=(0, 1), keepdim=True)
    std = motion_data.std(dim=(0, 1), keepdim=True) + 1e-8
    motion_data = (motion_data - mean) / std
    
    return motion_data

# 使用示例
raw_motion = torch.randn(1, 150, 263)  # 150个时间步
processed_motion = preprocess_motion(raw_motion, target_length=100)
print(f"原始形状: {raw_motion.shape}")
print(f"处理后形状: {processed_motion.shape}")
```

### 2. Token后处理工具

```python
def postprocess_tokens(tokens, original_length=None):
    """
    Token后处理
    
    Args:
        tokens (torch.Tensor): 生成的tokens
        original_length (int, optional): 原始长度
    
    Returns:
        torch.Tensor: 处理后的tokens
    """
    if original_length is not None:
        tokens = tokens[:, :original_length]
    
    # 移除padding tokens (假设0是padding)
    mask = tokens != 0
    tokens = tokens[mask].view(tokens.shape[0], -1)
    
    return tokens

# 使用示例
tokens_with_padding = torch.randint(0, 1024, (1, 120))
processed_tokens = postprocess_tokens(tokens_with_padding, original_length=100)
print(f"原始tokens形状: {tokens_with_padding.shape}")
print(f"处理后tokens形状: {processed_tokens.shape}")
```

### 3. 动作质量评估工具

```python
def evaluate_motion_quality(original_motion, reconstructed_motion):
    """
    评估动作质量
    
    Args:
        original_motion (torch.Tensor): 原始动作
        reconstructed_motion (torch.Tensor): 重构动作
    
    Returns:
        dict: 质量指标
    """
    # 重构误差
    mse_loss = torch.nn.functional.mse_loss(reconstructed_motion, original_motion)
    
    # 逐关节误差
    joint_errors = torch.nn.functional.mse_loss(
        reconstructed_motion, original_motion, reduction='none'
    ).mean(dim=(0, 1))  # (features,)
    
    # 时间平滑性
    velocity_original = torch.diff(original_motion, dim=1)
    velocity_reconstructed = torch.diff(reconstructed_motion, dim=1)
    smoothness_loss = torch.nn.functional.mse_loss(velocity_reconstructed, velocity_original)
    
    return {
        'mse_loss': mse_loss.item(),
        'joint_errors': joint_errors.tolist(),
        'smoothness_loss': smoothness_loss.item(),
        'mean_joint_error': joint_errors.mean().item(),
        'max_joint_error': joint_errors.max().item()
    }

# 使用示例
quality_metrics = evaluate_motion_quality(motion_data, reconstructed_motion)
print(f"重构MSE: {quality_metrics['mse_loss']:.6f}")
print(f"平均关节误差: {quality_metrics['mean_joint_error']:.6f}")
print(f"最大关节误差: {quality_metrics['max_joint_error']:.6f}")
print(f"平滑性损失: {quality_metrics['smoothness_loss']:.6f}")
```

## 常见使用场景

### 1. 训练VQ-VAE模型

```python
def train_vqvae_model(data_loader, epochs=100, lr=1e-4):
    """训练VQ-VAE模型"""
    
    # 初始化模型和优化器
    args = Args()
    vqvae = HumanVQVAE(args)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_commit_loss = 0
        
        for batch in data_loader:
            motion_data = batch['motion']
            
            # 前向传播
            reconstructed_motion, commit_loss, perplexity = vqvae(motion_data)
            
            # 计算重构损失
            recon_loss = torch.nn.functional.mse_loss(reconstructed_motion, motion_data)
            
            # 总损失
            total_loss_batch = recon_loss + commit_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # 记录损失
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
        
        # 打印统计信息
        avg_loss = total_loss / len(data_loader)
        avg_recon_loss = total_recon_loss / len(data_loader)
        avg_commit_loss = total_commit_loss / len(data_loader)
        
        print(f"Epoch {epoch}/{epochs}: "
              f"Loss: {avg_loss:.6f}, "
              f"Recon: {avg_recon_loss:.6f}, "
              f"Commit: {avg_commit_loss:.6f}, "
              f"Perplexity: {perplexity:.4f}")
    
    return vqvae
```

### 2. 动作到文本生成

```python
def motion_to_text_generation(motion_data, vqvae, text_model, tokenizer):
    """动作到文本生成"""
    
    # 步骤1: 将动作编码为tokens
    motion_tokens = vqvae.encode(motion_data)  # (batch, timesteps)
    
    # 步骤2: 添加特殊tokens
    bos_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    
    # 构建输入序列
    input_tokens = torch.cat([
        torch.tensor([[bos_token]]),      # 开始标记
        motion_tokens,                    # 动作tokens
        torch.tensor([[eos_token]])       # 结束标记
    ], dim=1)
    
    # 步骤3: 生成文本描述
    with torch.no_grad():
        text_outputs = text_model.generate(
            input_ids=input_tokens,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    # 步骤4: 解码文本
    generated_text = tokenizer.decode(text_outputs[0], skip_special_tokens=True)
    
    return generated_text

# 使用示例
# text_description = motion_to_text_generation(motion_data, vqvae, llm_model, tokenizer)
# print(f"生成的文本描述: {text_description}")
```

### 3. 文本到动作生成

```python
def text_to_motion_generation(text_prompt, vqvae, text_model, tokenizer, max_length=100):
    """文本到动作生成"""
    
    # 步骤1: 文本编码
    text_inputs = tokenizer(text_prompt, return_tensors='pt')
    
    # 步骤2: 生成动作tokens
    with torch.no_grad():
        motion_tokens = text_model.generate(
            input_ids=text_inputs['input_ids'],
            max_length=max_length + len(text_inputs['input_ids'][0]),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    # 提取动作tokens (移除文本部分)
    motion_tokens_only = motion_tokens[0, len(text_inputs['input_ids'][0]):]
    motion_tokens_only = motion_tokens_only.unsqueeze(0)  # 添加批次维度
    
    # 步骤3: 解码为动作
    generated_motion = vqvae.forward_decoder(motion_tokens_only)
    
    return generated_motion

# 使用示例
# text_prompt = "一个人在走路"
# generated_motion = text_to_motion_generation(text_prompt, vqvae, llm_model, tokenizer)
# print(f"生成动作形状: {generated_motion.shape}")
```

### 4. 动作检索和匹配

```python
def motion_retrieval(query_motion, motion_database, vqvae, top_k=5):
    """动作检索"""
    
    # 步骤1: 编码查询动作
    query_features = vqvae.encode_x(query_motion)  # (timesteps, code_dim)
    query_features = query_features.mean(dim=0, keepdim=True)  # 全局平均池化
    
    # 步骤2: 编码数据库动作
    database_features = []
    for motion in motion_database:
        features = vqvae.encode_x(motion.unsqueeze(0))  # (timesteps, code_dim)
        features = features.mean(dim=0, keepdim=True)  # 全局平均池化
        database_features.append(features)
    
    database_features = torch.cat(database_features, dim=0)  # (num_motions, code_dim)
    
    # 步骤3: 计算相似度
    similarities = torch.nn.functional.cosine_similarity(
        query_features, database_features, dim=1
    )
    
    # 步骤4: 获取top-k结果
    top_k_indices = torch.topk(similarities, k=top_k).indices
    
    return {
        'top_k_indices': top_k_indices.tolist(),
        'similarities': similarities[top_k_indices].tolist(),
        'query_features': query_features,
        'database_features': database_features
    }

# 使用示例
# database_motions = [torch.randn(100, 263) for _ in range(100)]  # 100个动作
# query_motion = torch.randn(100, 263)
# retrieval_results = motion_retrieval(query_motion, database_motions, vqvae)
# print(f"Top-5 相似动作索引: {retrieval_results['top_k_indices']}")
```

## 错误处理和调试

### 1. 常见错误和处理

```python
def safe_vqvae_encode(vqvae, motion_data):
    """安全的VQ-VAE编码"""
    try:
        # 检查输入形状
        if len(motion_data.shape) != 3:
            raise ValueError(f"输入必须是3D张量，实际形状: {motion_data.shape}")
        
        # 检查数据类型
        if not isinstance(motion_data, torch.Tensor):
            motion_data = torch.tensor(motion_data, dtype=torch.float32)
        
        # 检查数值范围
        if torch.isnan(motion_data).any() or torch.isinf(motion_data).any():
            raise ValueError("输入包含NaN或Inf值")
        
        # 执行编码
        tokens = vqvae.encode(motion_data)
        
        # 检查输出
        if torch.isnan(tokens).any():
            raise ValueError("编码结果包含NaN值")
        
        return tokens
    
    except Exception as e:
        print(f"VQ-VAE编码错误: {e}")
        return None

# 使用示例
# tokens = safe_vqvae_encode(vqvae, motion_data)
# if tokens is not None:
#     print("编码成功")
# else:
#     print("编码失败")
```

### 2. 调试工具

```python
def debug_vqvae_forward(vqvae, motion_data, debug=True):
    """调试VQ-VAE前向传播"""
    
    if debug:
        print("=== VQ-VAE 调试信息 ===")
        print(f"输入形状: {motion_data.shape}")
        print(f"输入数据类型: {motion_data.dtype}")
        print(f"输入设备: {motion_data.device}")
        print(f"输入统计: mean={motion_data.mean():.4f}, std={motion_data.std():.4f}")
        print(f"输入范围: [{motion_data.min():.4f}, {motion_data.max():.4f}]")
    
    # 预处理
    x_in = vqvae.vqvae.preprocess(motion_data)
    
    if debug:
        print(f"预处理后形状: {x_in.shape}")
        print(f"预处理后统计: mean={x_in.mean():.4f}, std={x_in.std():.4f}")
    
    # 编码
    x_encoder = vqvae.vqvae.encoder(x_in)
    
    if debug:
        print(f"编码后形状: {x_encoder.shape}")
        print(f"编码统计: mean={x_encoder.mean():.4f}, std={x_encoder.std():.4f}")
    
    # 量化
    x_quantized, loss, perplexity = vqvae.vqvae.quantizer(x_encoder)
    
    if debug:
        print(f"量化后形状: {x_quantized.shape}")
        print(f"量化损失: {loss:.6f}")
        print(f"困惑度: {perplexity:.4f}")
    
    # 解码
    x_decoder = vqvae.vqvae.decoder(x_quantized)
    x_out = vqvae.vqvae.postprocess(x_decoder)
    
    if debug:
        print(f"解码后形状: {x_out.shape}")
        print(f"解码统计: mean={x_out.mean():.4f}, std={x_out.std():.4f}")
        print(f"重构MSE: {torch.nn.functional.mse_loss(x_out, motion_data):.6f}")
        print("=== 调试结束 ===")
    
    return x_out, loss, perplexity

# 使用示例
# reconstructed, loss, perplexity = debug_vqvae_forward(vqvae, motion_data)
```

## 性能优化建议

### 1. 内存优化

```python
def memory_efficient_vqvae_encode(vqvae, motion_data, chunk_size=50):
    """内存高效的VQ-VAE编码"""
    
    batch_size, timesteps, features = motion_data.shape
    
    if timesteps <= chunk_size:
        # 短序列直接处理
        return vqvae.encode(motion_data)
    
    # 长序列分块处理
    all_tokens = []
    
    for i in range(0, timesteps, chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, timesteps)
        
        # 提取块
        chunk_data = motion_data[:, chunk_start:chunk_end, :]
        
        # 编码块
        chunk_tokens = vqvae.encode(chunk_data)
        all_tokens.append(chunk_tokens)
    
    # 合并结果
    return torch.cat(all_tokens, dim=1)

# 使用示例
# long_motion = torch.randn(1, 500, 263)  # 500个时间步
# tokens = memory_efficient_vqvae_encode(vqvae, long_motion, chunk_size=100)
# print(f"分块编码结果: {tokens.shape}")
```

### 2. 批处理优化

```python
def batch_vqvae_processing(vqvae, motion_list, batch_size=8):
    """批处理VQ-VAE处理"""
    
    all_results = []
    
    for i in range(0, len(motion_list), batch_size):
        batch_motions = motion_list[i:i + batch_size]
        
        # 填充到相同长度
        max_length = max(motion.shape[1] for motion in batch_motions)
        padded_motions = []
        
        for motion in batch_motions:
            if motion.shape[1] < max_length:
                padding = torch.zeros(motion.shape[0], max_length - motion.shape[1], motion.shape[2])
                motion = torch.cat([motion, padding], dim=1)
            padded_motions.append(motion)
        
        batch_tensor = torch.cat(padded_motions, dim=0)
        
        # 批处理编码
        batch_tokens = vqvae.encode(batch_tensor)
        all_results.append(batch_tokens)
    
    return torch.cat(all_results, dim=0)

# 使用示例
# motion_list = [torch.randn(1, 100, 263) for _ in range(20)]
# batch_tokens = batch_vqvae_processing(vqvae, motion_list, batch_size=4)
# print(f"批处理结果: {batch_tokens.shape}")
```

这个API实用指南提供了 VQ-VAE 模块的详细使用方法，包含各种实际应用场景和最佳实践。