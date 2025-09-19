# MotionLLM 使用示例和最佳实践

## 实际使用场景示例

### 场景1：视频问答系统

#### 完整的设置流程

```bash
# 1. 环境准备
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 2. 检查模型文件
ls -la ./checkpoints/vicuna-7b-v1.5/
# 应该看到: lit_model.pth, tokenizer.model

ls -la ./finetuned_models/
# 应该看到: video_qa_lora.pth, video_mlp_projector.pth

# 3. 运行视频问答
python CLI.py \
    --lora_path ./finetuned_models/video_qa_lora.pth \
    --mlp_path ./finetuned_models/video_mlp_projector.pth \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 200
```

#### 交互式问答示例

```bash
# 程序启动后，按照提示输入：

# 第一步：输入视频路径
Input video path: ./examples/walking_person.mp4

# 第二步：输入问题
Your question: 视频中的人在做什么动作？

# 模型输出
================================
Model output: 视频中一个人正在走路。他迈着有节奏的步伐，双臂自然摆动，看起来是在公园或人行道上行走。
================================

# 继续提问同一视频
Your question: 他走路的速度如何？

================================
Model output: 从视频中可以看出，这个人的走路速度适中，不是很快也不是很慢，是一种正常的步行速度。
================================
```

### 场景2：体育动作分析

#### 专门的体育分析配置

```python
# sports_analysis_config.py
class SportsAnalysisConfig:
    def __init__(self):
        # 专门针对体育视频的配置
        self.lora_path = "./finetuned_models/sports_analysis_lora.pth"
        self.mlp_path = "./finetuned_models/sports_mlp_projector.pth"
        
        # 体育分析优化参数
        self.max_new_tokens = 300  # 更长的详细描述
        self.temperature = 0.6     # 更保守的采样
        self.top_k = 100           # 更小的候选集
        
        # 多模态配置
        self.video_tower = "LanguageBind/LanguageBind_Video_merge"
        self.mm_projector_type = "mlp2x_gelu"
        self.mm_hidden_size = 1024
        self.hidden_size = 4096

# 使用示例
config = SportsAnalysisConfig()

# 模拟体育分析问答
def analyze_sports_video(video_path, questions):
    """
    分析体育视频并回答相关问题
    """
    print(f"分析体育视频: {video_path}")
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        
        # 模拟模型输出
        if "投篮" in question:
            answer = "从视频中可以看到，运动员正在进行篮球投篮动作。他的投篮姿势标准，手腕动作正确，球出手后旋转良好。这是一个成功的跳投动作。"
        elif "跑步" in question:
            answer = "运动员的跑步姿势很好，步频稳定，手臂摆动协调。从技术动作来看，这是一个专业的跑步姿势，步幅和步频都很合理。"
        else:
            answer = "视频中展示了专业的体育动作，运动员的技术动作标准，符合专业运动的要求。"
        
        print(f"分析结果: {answer}")
```

#### 体育分析实际对话

```bash
# 运行体育分析系统
python CLI.py \
    --lora_path ./finetuned_models/sports_analysis_lora.pth \
    --mlp_path ./finetuned_models/sports_mlp_projector.pth \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --max_new_tokens 300 \
    --temperature 0.6 \
    --top_k 100

# 交互对话
Input video path: ./examples/basketball_shooting.mp4

Your question: 这个篮球投篮动作标准吗？
================================
Model output: 这个投篮动作相当标准。运动员的起跳姿势正确，手臂伸直，手腕下压动作到位，球出手后有很好的后旋。不过可以注意到落地时膝盖弯曲不够充分，建议加强落地缓冲训练。
================================

Your question: 投篮的角度大概是多少？
================================
Model output: 从视频分析来看，投篮角度大约在45-50度之间，这是一个理想的投篮角度。这种角度既保证了足够的弧度，又能够提高投篮命中率。球的飞行轨迹看起来很标准。
================================
```

### 场景3：医疗康复动作评估

#### 医疗康复专用配置

```python
# medical_rehabilitation_config.py
class MedicalRehabilitationConfig:
    def __init__(self):
        # 医疗康复专用模型
        self.lora_path = "./finetuned_models/medical_rehab_lora.pth"
        self.mlp_path = "./finetuned_models/medical_mlp_projector.pth"
        
        # 医疗分析参数
        self.max_new_tokens = 250
        self.temperature = 0.4     # 更保守，更确定
        self.top_k = 50            # 更小的候选集
        
        # 重点关注动作的精确性
        self.mm_projector_type = "qformer2_64"  # 使用Q-Former提取更精确的特征
```

#### 康复评估对话示例

```bash
# 运行康复评估系统
python CLI.py \
    --lora_path ./finetuned_models/medical_rehab_lora.pth \
    --mlp_path ./finetuned_models/medical_mlp_projector.pth \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type qformer2_64 \
    --max_new_tokens 250 \
    --temperature 0.4 \
    --top_k 50

# 康复评估对话
Input video path: ./examples/knee_rehabilitation.mp4

Your question: 这个膝关节康复动作做得正确吗？
================================
Model output: 从康复医学的角度来看，这个膝关节屈伸动作基本正确。患者的动作速度适中，关节活动范围在安全范围内。不过需要注意的是，动作过程中有轻微的膝关节内翻，建议在治疗师指导下进行纠正。整体来说，这是一个适合该康复阶段的动作。
================================
```

## 最佳实践指南

### 1. 模型选择和配置

#### 根据任务选择合适的投影器

```python
def choose_projector_by_task(task_type):
    """根据任务类型选择投影器"""
    
    task_projector_map = {
        'general_qa': 'mlp2x_gelu',        # 通用问答
        'detailed_analysis': 'qformer2_64', # 详细分析
        'quick_response': 'linear',        # 快速响应
        'high_accuracy': 'mlp3x_gelu',     # 高精度
        'memory_efficient': 'identity'     # 内存高效
    }
    
    return task_projector_map.get(task_type, 'mlp2x_gelu')

# 使用示例
projector_type = choose_projector_by_task('sports_analysis')
print(f"推荐的投影器类型: {projector_type}")
```

#### LoRA参数调优策略

```python
def optimize_lora_parameters(model_size, task_complexity):
    """优化LoRA参数"""
    
    # 基础配置
    base_configs = {
        '7B': {'r': 64, 'alpha': 16, 'dropout': 0.05},
        '13B': {'r': 32, 'alpha': 8, 'dropout': 0.1},
        '30B': {'r': 16, 'alpha': 4, 'dropout': 0.1}
    }
    
    # 根据任务复杂度调整
    complexity_multipliers = {
        'simple': 0.5,
        'medium': 1.0,
        'complex': 1.5
    }
    
    base_config = base_configs.get(model_size, base_configs['7B'])
    multiplier = complexity_multipliers.get(task_complexity, 1.0)
    
    optimized_config = {
        'lora_r': int(base_config['r'] * multiplier),
        'lora_alpha': int(base_config['alpha'] * multiplier),
        'lora_dropout': min(base_config['dropout'] * (2 - multiplier), 0.3)
    }
    
    return optimized_config

# 使用示例
lora_config = optimize_lora_parameters('7B', 'complex')
print(f"优化后的LoRA配置: {lora_config}")
```

### 2. 数据预处理最佳实践

#### 视频预处理优化

```python
def optimize_video_processing(video_path, target_frames=8):
    """优化视频预处理"""
    
    import cv2
    import numpy as np
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 智能帧采样策略
    if total_frames <= target_frames:
        # 如果视频较短，使用所有帧
        frame_indices = list(range(total_frames))
    else:
        # 如果视频较长，均匀采样
        step = total_frames // target_frames
        frame_indices = [i * step for i in range(target_frames)]
    
    # 提取帧
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小和标准化
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    return np.array(frames)

# 使用示例
video_frames = optimize_video_processing("./examples/sample_video.mp4")
print(f"处理后的视频帧形状: {video_frames.shape}")
```

#### 批量处理优化

```python
def batch_video_processing(video_paths, batch_size=4):
    """批量处理多个视频"""
    
    from torch.utils.data import Dataset, DataLoader
    
    class VideoDataset(Dataset):
        def __init__(self, video_paths):
            self.video_paths = video_paths
        
        def __len__(self):
            return len(self.video_paths)
        
        def __getitem__(self, idx):
            return optimize_video_processing(self.video_paths[idx])
    
    # 创建数据加载器
    dataset = VideoDataset(video_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 批量处理
    results = []
    for batch in dataloader:
        # 这里可以添加模型推理逻辑
        results.append(batch.shape)
    
    return results

# 使用示例
video_list = ["./examples/video1.mp4", "./examples/video2.mp4"]
batch_results = batch_video_processing(video_list)
print(f"批量处理结果: {batch_results}")
```

### 3. 性能优化技巧

#### 内存管理优化

```python
def optimize_memory_usage(model, input_size, device='cuda'):
    """优化内存使用"""
    
    import torch
    
    # 1. 使用半精度
    if device == 'cuda':
        model = model.half()
    
    # 2. 清理缓存
    torch.cuda.empty_cache()
    
    # 3. 分块处理大输入
    def process_large_input(model, input_data, chunk_size=8):
        results = []
        for i in range(0, len(input_data), chunk_size):
            chunk = input_data[i:i+chunk_size]
            with torch.no_grad():
                chunk_result = model(chunk)
            results.append(chunk_result)
            torch.cuda.empty_cache()  # 处理完每个块后清理内存
        return torch.cat(results, dim=0)
    
    return process_large_input

# 使用示例
def memory_efficient_inference(model, video_features):
    """内存高效的推理"""
    
    # 检查输入大小
    if video_features.shape[1] > 1000:  # 如果序列很长
        print("检测到大序列输入，使用分块处理...")
        processor = optimize_memory_usage(model, video_features.shape)
        return processor(model, video_features)
    else:
        with torch.no_grad():
            return model(video_features)
```

#### 推理速度优化

```python
def optimize_inference_speed(model, optimize_level='medium'):
    """优化推理速度"""
    
    import torch
    
    if optimize_level == 'basic':
        # 基本优化
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    elif optimize_level == 'medium':
        # 中等优化
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # 使用torch.jit优化
        try:
            model = torch.jit.script(model)
        except:
            print("JIT优化失败，使用eval模式")
    
    elif optimize_level == 'advanced':
        # 高级优化
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # 使用半精度和编译优化
        if torch.cuda.is_available():
            model = model.half()
        
        try:
            model = torch.compile(model)
        except:
            print("编译优化失败，使用基本优化")
    
    return model

# 使用示例
optimized_model = optimize_inference_speed(model, optimize_level='advanced')
```

### 4. 错误处理和调试

#### 常见错误处理

```python
def handle_common_errors(error_type, additional_info=None):
    """处理常见错误"""
    
    error_solutions = {
        'cuda_out_of_memory': [
            "减少批次大小",
            "使用更小的序列长度",
            "使用模型量化",
            "清理GPU缓存"
        ],
        'model_load_failed': [
            "检查模型文件路径",
            "验证模型文件完整性",
            "检查文件权限",
            "确认PyTorch版本兼容性"
        ],
        'video_processing_error': [
            "检查视频格式支持",
            "验证视频文件完整性",
            "安装必要的编解码器",
            "检查视频文件权限"
        ],
        'tokenization_error': [
            "检查分词器路径",
            "验证文本编码",
            "检查特殊字符处理",
            "确认文本长度限制"
        ]
    }
    
    if error_type in error_solutions:
        print(f"错误类型: {error_type}")
        print("解决方案:")
        for i, solution in enumerate(error_solutions[error_type], 1):
            print(f"  {i}. {solution}")
        
        if additional_info:
            print(f"附加信息: {additional_info}")
    else:
        print(f"未知错误类型: {error_type}")
        print("请检查日志文件或联系技术支持")

# 使用示例
try:
    # 可能出现错误的代码
    result = model.process_large_video(video_path)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        handle_common_errors('cuda_out_of_memory', f"视频路径: {video_path}")
    else:
        handle_common_errors('unknown', str(e))
```

#### 调试工具

```python
class ModelDebugger:
    """模型调试工具"""
    
    def __init__(self, model):
        self.model = model
    
    def check_model_dimensions(self):
        """检查模型维度一致性"""
        print("=== 模型维度检查 ===")
        
        # 检查投影器
        if hasattr(self.model, 'mm_projector'):
            projector = self.model.mm_projector
            print(f"投影器类型: {type(projector).__name__}")
            
            # 检查参数
            total_params = sum(p.numel() for p in projector.parameters())
            print(f"投影器参数量: {total_params:,}")
        
        # 检查编码器
        if hasattr(self.model, 'get_image_tower'):
            image_tower = self.model.get_image_tower()
            if image_tower:
                print(f"图像编码器: {type(image_tower).__name__}")
        
        if hasattr(self.model, 'get_video_tower'):
            video_tower = self.model.get_video_tower()
            if video_tower:
                print(f"视频编码器: {type(video_tower).__name__}")
    
    def test_forward_pass(self, test_input):
        """测试前向传播"""
        print("=== 前向传播测试 ===")
        
        try:
            with torch.no_grad():
                output = self.model(test_input)
            
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"输出统计: mean={output.mean():.4f}, std={output.std():.4f}")
            print("✓ 前向传播测试通过")
            
            return True
        except Exception as e:
            print(f"✗ 前向传播测试失败: {e}")
            return False
    
    def profile_memory_usage(self, test_input):
        """分析内存使用"""
        print("=== 内存使用分析 ===")
        
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # 测试前内存
            mem_before = torch.cuda.memory_allocated() / 1024**3
            
            # 运行测试
            with torch.no_grad():
                _ = self.model(test_input)
            
            # 测试后内存
            mem_after = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"测试前内存: {mem_before:.2f} GB")
            print(f"测试后内存: {mem_after:.2f} GB")
            print(f"峰值内存: {peak_memory:.2f} GB")
            print(f"内存增量: {peak_memory - mem_before:.2f} GB")
        else:
            print("CUDA不可用，无法分析GPU内存使用")

# 使用示例
debugger = ModelDebugger(model)
debugger.check_model_dimensions()
debugger.test_forward_pass(test_input)
debugger.profile_memory_usage(test_input)
```

### 5. 部署和集成

#### Web服务部署

```python
# web_service.py
from flask import Flask, request, jsonify
import torch
import tempfile
import os

app = Flask(__name__)

class MotionLLMService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    def load_model(self, config):
        """加载模型"""
        # 这里集成CLI.py中的模型加载逻辑
        print("加载MotionLLM模型...")
        # 模型加载代码
        print("模型加载完成")

service = MotionLLMService()

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """视频分析API"""
    
    try:
        # 获取视频文件
        video_file = request.files['video']
        question = request.form.get('question', '')
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name
        
        # 调用模型分析
        result = service.analyze_video(video_path, question)
        
        # 清理临时文件
        os.unlink(video_path)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    service.load_model(config)
    app.run(host='0.0.0.0', port=5000)
```

#### Docker部署配置

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制模型和代码
COPY checkpoints/ ./checkpoints/
COPY finetuned_models/ ./finetuned_models/
COPY CLI.py .
COPY options/ ./options/
COPY models/ ./models/

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "web_service.py"]
```

通过这些使用示例和最佳实践，您可以更好地利用MotionLLM进行各种多模态任务，并根据具体需求进行优化和定制。