# MotionLLM 核心模块实现细节

## 1. 多模态编码器实现细节

### 1.1 LanguageBind 编码器架构

#### 核心实现 (`models/multimodal_encoder/languagebind/`)

**统一的模态接口**:
```python
# languagebind/video/modeling_video.py
class LanguageBindVideoTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 使用 CLIP 作为基础架构
        self.vision_model = CLIPVisionTransformer(config)
        
    def forward(self, pixel_values, *args, **kwargs):
        # 输入: (batch, channels, frames, height, width)
        # 输出: (batch, frames * sequence_length, hidden_size)
        batch_size, channels, num_frames, height, width = pixel_values.shape
        
        # 重塑为 (batch * frames, channels, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        pixel_values = pixel_values.view(batch_size * num_frames, channels, height, width)
        
        # CLIP 编码
        vision_outputs = self.vision_model(pixel_values)
        
        # 返回拼接的特征
        return vision_outputs.last_hidden_state
```

**预处理器集成**:
```python
# languagebind/video/processing_video.py
class LanguageBindVideoProcessor:
    def __init__(self, config):
        self.image_processor = CLIPImageProcessor.from_pretrained(config)
        self.video_processor = VideoProcessor(config)
    
    def __call__(self, videos, *args, **kwargs):
        # 视频采样和归一化
        processed_videos = self.video_processor(videos)
        return {"pixel_values": processed_videos}
```

#### 多模态统一处理

**模态注册机制**:
```python
# languagebind/__init__.py
LANGUAGEBIND_TOWERS = {
    'image': LanguageBindImageTower,
    'video': LanguageBindVideoTower,
    'audio': LanguageBindAudioTower,
    'thermal': LanguageBindThermalTower,
    'depth': LanguageBindDepthTower
}
```

### 1.2 CLIP 编码器实现

#### 标准化接口 (`models/multimodal_encoder/clip_encoder.py`)

**CLIPVisionTower 类**:
```python
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        # 从 Hugging Face 加载 CLIP 模型
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)  # 冻结梯度
        
        # 特征选择层
        self.select_feature = getattr(self, f"select_{self.select_layer}", None)
        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        # 选择指定层的特征
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features
    
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        
        # 标准 CLIP 前向传播
        image_forward_outs = self.vision_tower(images, output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs)
        
        return image_features
```

### 1.3 MAE 编码器实现

#### 掩码自编码器 (`models/multimodal_encoder/mae_encoder.py`)

**MAEVisionTower 类**:
```python
class MAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        # 加载 MAE 预训练权重
        self.vision_tower = AutoModelForMaskedImageModeling.from_pretrained(self.vision_tower_name)
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        
    def forward(self, images):
        # MAE 特征提取
        with torch.no_grad():
            outputs = self.vision_tower(images)
            features = outputs.last_hidden_state
        
        return features
```

## 2. 多模态投影器实现细节

### 2.1 MLP 投影器实现

#### 多层感知机架构 (`models/multimodal_projector/builder.py`)

**MLP 构建逻辑**:
```python
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    elif projector_type.startswith('mlp') and projector_type.endswith('gelu'):
        # 解析 MLP 深度: mlp2x_gelu -> depth=2
        mlp_depth = int(projector_type.split('x')[0][3:])
        
        modules = []
        # 第一层: mm_hidden_size -> hidden_size
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        
        # 中间层: 保持 hidden_size 维度
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        return nn.Sequential(*modules)
    
    # ... 其他投影器类型
```

#### 具体实现示例
```python
# mlp2x_gelu 的具体结构
mlp_projector = nn.Sequential(
    nn.Linear(1024, 4096),  # LanguageBind -> Vicuna
    nn.GELU(),
    nn.Linear(4096, 4096)
)
```

### 2.2 Q-Former 投影器实现

#### BLIP-2 架构适配 (`models/multimodal_projector/builder.py`)

**Blip2Model 类**:
```python
class Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)
        
        # 可学习的查询 tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        
        # Q-Former 模型
        self.qformer = Blip2QFormerModel(config.qformer_config)
        
        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(self, pixel_values):
        # 输入已经是编码后的特征
        image_embeds = pixel_values
        
        # 构建注意力掩码
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], 
            dtype=torch.long, 
            device=image_embeds.device
        )
        
        # 扩展查询 tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        # Q-Former 处理
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        ).last_hidden_state
        
        # 最终投影
        return self.proj(query_outputs)
```

#### Q-Former 配置生成
```python
def qformer_config_template(config, projector_type):
    # 解析配置: qformer2_64 -> 2层, 64个查询
    match = re.search(r"qformer(\d+)_(\d+)", projector_type)
    num_hidden_layers = int(match.group(1))
    num_query_tokens = int(match.group(2))
    
    # 动态生成 Q-Former 配置
    qformer_config = type('Blip2Config', (PretrainedConfig,), {
        "num_query_tokens": num_query_tokens,
        "hidden_size": config.hidden_size,
        "mm_hidden_size": config.mm_hidden_size,
        "qformer_config": type('qformer_config', (PretrainedConfig,), {
            "hidden_size": config.mm_hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 32,
            "intermediate_size": config.mm_hidden_size * 4,
            # ... 其他配置
        })()
    })()
    
    return qformer_config
```

## 3. VQ-VAE 实现细节

### 3.1 编码器-解码器架构

#### 编码器实现 (`models/encdec.py`)

**Encoder 类**:
```python
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3, 
                 stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                 activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # 初始卷积
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # 下采样块
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, 
                        activation=activation, norm=norm),
            )
            blocks.append(block)
        
        # 输出投影
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)
```

#### 解码器实现
```python
class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=3,
                 stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                 activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        
        # 输入投影
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # 上采样块
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, 
                        reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        
        # 最终输出层
        blocks.extend([
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        ])
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)
```

### 3.2 ResNet1D 组件实现

#### ResConv1DBlock (`models/resnet.py`)
```python
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', 
                 norm=None, dropout=None):
        super().__init__()
        
        # 归一化层选择
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(32, n_in, eps=1e-6)
            self.norm2 = nn.GroupNorm(32, n_in, eps=1e-6)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_in, eps=1e-6)
            self.norm2 = nn.BatchNorm1d(n_in, eps=1e-6)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # 激活函数选择
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()  # Swish
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        
        # 卷积层
        padding = dilation
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
    
    def forward(self, x):
        x_orig = x
        
        # 第一个卷积块
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
        
        x = self.conv1(x)
        
        # 第二个卷积块
        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        
        x = self.conv2(x)
        
        # 残差连接
        return x + x_orig
```

#### Resnet1D 组合
```python
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, 
                 reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        # 构建 ResNet 块
        blocks = [
            ResConv1DBlock(
                n_in, n_in, 
                dilation=dilation_growth_rate ** depth,
                activation=activation, norm=norm
            ) 
            for depth in range(n_depth)
        ]
        
        # 可选的反向膨胀率
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)
```

### 3.3 量化器实现细节

#### EMA 重置量化器 (`models/quantize_cnn.py`)

**QuantizeEMAReset 类**:
```python
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu  # EMA 动量
        self.reset_codebook()
    
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        # 注册 codebook 为 buffer
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())
    
    def _tile(self, x):
        # 如果样本数少于 codebook 大小，进行平铺
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    def init_codebook(self, x):
        # 初始化 codebook
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
    
    def quantize(self, x):
        # 计算距离并找到最近的 code
        k_w = self.codebook.t()
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True) - 
            2 * torch.matmul(x, k_w) + 
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx
    
    def dequantize(self, code_idx):
        # 从 codebook 中解码
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def forward(self, x):
        N, width, T = x.shape
        
        # 预处理: NCT -> NTC -> [NT, C]
        x = self.preprocess(x)
        
        # 初始化 codebook
        if self.training and not self.init:
            self.init_codebook(x)
        
        # 量化和反量化
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        
        # 更新 codebook
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        # 损失计算
        commit_loss = F.mse_loss(x, x_d.detach())
        
        # 直通估计器
        x_d = x + (x_d - x).detach()
        
        # 后处理: [NT, C] -> NTC -> NCT
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        
        return x_d, commit_loss, perplexity
```

#### Codebook 更新机制
```python
@torch.no_grad()
def update_codebook(self, x, code_idx):
    # 计算 one-hot 编码
    code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
    code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
    
    # 计算统计量
    code_sum = torch.matmul(code_onehot, x)
    code_count = code_onehot.sum(dim=-1)
    
    # 准备随机初始化的 code
    out = self._tile(x)
    code_rand = out[:self.nb_code]
    
    # EMA 更新
    self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
    self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count
    
    # 使用掩码决定是否更新
    usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
    code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
    
    # 更新 codebook
    self.codebook = usage * code_update + (1 - usage) * code_rand
    
    # 计算困惑度
    prob = code_count / torch.sum(code_count)
    perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
    
    return perplexity
```

## 4. 运动编码器实现细节

### 4.1 运动卷积编码器 (`models/modules.py`)

**MovementConvEncoder 类**:
```python
class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        
        # 主卷积网络
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 输出投影
        self.out_net = nn.Linear(output_size, output_size)
        
        # 权重初始化
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)
    
    def forward(self, inputs):
        # 输入形状变化: (batch, seq_len, features) -> (batch, features, seq_len)
        inputs = inputs.permute(0, 2, 1)
        
        # 卷积编码
        outputs = self.main(inputs)
        
        # 恢复形状
        outputs = outputs.permute(0, 2, 1)
        
        # 最终投影
        return self.out_net(outputs)
```

### 4.2 双向 GRU 编码器

**TextEncoderBiGRUCo 类**:
```python
class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device
        
        # 位置编码
        self.pos_emb = nn.Linear(pos_size, word_size)
        
        # 输入嵌入
        self.input_emb = nn.Linear(word_size, hidden_size)
        
        # 双向 GRU
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 输出网络
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 双向拼接
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        
        # 权重初始化
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        
        # 隐藏状态
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))
    
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        
        # 位置编码
        pos_embs = self.pos_emb(pos_onehot)
        
        # 词嵌入 + 位置编码
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        
        # 初始化隐藏状态
        hidden = self.hidden.repeat(1, num_samples, 1)
        
        # 处理变长序列
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        
        # GRU 编码
        gru_seq, gru_last = self.gru(emb, hidden)
        
        # 拼接双向输出
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        
        return self.output_net(gru_last)
```

**MotionEncoderBiGRUCo 类**:
```python
class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device
        
        # 输入嵌入
        self.input_emb = nn.Linear(input_size, hidden_size)
        
        # 双向 GRU
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 输出网络
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        
        # 权重初始化
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        
        # 隐藏状态
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))
    
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]
        
        # 输入嵌入
        input_embs = self.input_emb(inputs)
        
        # 初始化隐藏状态
        hidden = self.hidden.repeat(1, num_samples, 1)
        
        # 处理变长序列
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)
        
        # GRU 编码
        gru_seq, gru_last = self.gru(emb, hidden)
        
        # 拼接双向输出
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        
        return self.output_net(gru_last)
```

## 5. 评估包装器实现细节

### 5.1 模型构建 (`models/evaluator_wrapper.py`)

**build_models 函数**:
```python
def build_models(opt):
    # 运动编码器
    movement_enc = MovementConvEncoder(
        opt.dim_pose - 4,          # 移除根关节旋转
        opt.dim_movement_enc_hidden,
        opt.dim_movement_latent
    )
    
    # 文本编码器
    text_enc = TextEncoderBiGRUCo(
        word_size=opt.dim_word,
        pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device
    )
    
    # 运动编码器
    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device
    )
    
    # 加载预训练权重
    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
        map_location=opt.device
    )
    
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    
    return text_enc, motion_enc, movement_enc
```

**EvaluatorModelWrapper 类**:
```python
class EvaluatorModelWrapper(object):
    def __init__(self, opt):
        # 数据集配置
        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')
        
        # 默认配置
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        
        # 构建模型
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device
        
        # 移动到设备并设置为评估模式
        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)
        
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
    
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            # 数据准备
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            
            # 运动编码
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
            
            # 文本编码
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        
        return text_embedding, motion_embedding
    
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            
            # 按长度排序以提高效率
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            # 运动编码
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        
        return motion_embedding
```

这些实现细节展示了 MotionLLM 各个模块的具体实现方式，包括网络架构设计、数据处理流程、以及模块间的接口定义。