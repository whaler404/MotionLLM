# MotionLLM 多模态投影器数据流示例和API使用指南

## 实际应用场景数据流

本章节通过具体的虚构数据示例，详细展示多模态投影器在不同任务场景下的数据流动过程。

### 场景1：图像描述生成任务

#### 任务描述
输入一张图片，生成对应的文本描述。需要将图像特征投影到语言模型特征空间。

#### 数据流示例

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# =========== 场景设置 ===========
# 假设我们有一个图像描述生成任务
# 输入：3张图片（猫、狗、鸟）
# 输出：对应的文本描述

class ImageDescriptionTask:
    def __init__(self):
        # 编码器配置
        self.clip_hidden_size = 768      # CLIP特征维度
        self.llm_hidden_size = 4096      # 语言模型特征维度
        self.batch_size = 3              # 3张图片
        
        # 投影器配置
        self.projector_types = ['linear', 'mlp2x_gelu']
        
        # 虚构数据：3张图片的特征
        self.image_descriptions = [
            "一只灰色的猫坐在沙发上",
            "一只棕色的狗在草地上奔跑", 
            "一只彩色的小鸟站在树枝上"
        ]
        
    def create_sample_image_features(self) -> torch.Tensor:
        """创建虚构的图像特征数据"""
        
        # 模拟3张图片的CLIP特征
        # 每张图片被分成196个patches，每个patch有768维特征
        image_features = torch.randn(self.batch_size, 196, self.clip_hidden_size)
        
        # 为不同类型的图片添加特定的"特征模式"
        # 猫的图片：在某些patch上添加"猫"的特征
        image_features[0, 50:70, :100] += 0.5  # 模拟猫的身体区域
        image_features[0, 80:100, 100:200] += 0.3  # 模拟猫的头部区域
        
        # 狗的图片：添加"狗"的特征
        image_features[1, 60:85, :150] += 0.4  # 模拟狗的身体
        image_features[1, 85:110, 150:250] += 0.35  # 模拟狗的头部
        
        # 鸟的图片：添加"鸟"的特征
        image_features[2, 40:55, :80] += 0.6  # 模拟鸟的身体
        image_features[2, 55:70, 80:160] += 0.45  # 模拟鸟的头部
        
        print("=== 图像特征数据生成 ===")
        print(f"生成图像特征形状: {image_features.shape}")
        print(f"图像描述: {self.image_descriptions}")
        
        for i, desc in enumerate(self.image_descriptions):
            feat_stats = {
                'mean': image_features[i].mean().item(),
                'std': image_features[i].std().item(),
                'max': image_features[i].max().item(),
                'min': image_features[i].min().item()
            }
            print(f"图片{i+1} ({desc[:10]}...): {feat_stats}")
        
        return image_features
    
    def create_projectors(self) -> Dict[str, nn.Module]:
        """创建不同类型的投影器"""
        
        projectors = {}
        
        # 线性投影器
        projectors['linear'] = nn.Linear(
            self.clip_hidden_size, 
            self.llm_hidden_size
        )
        
        # MLP投影器
        projectors['mlp2x_gelu'] = nn.Sequential(
            nn.Linear(self.clip_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        )
        
        print("\n=== 投影器创建 ===")
        for name, projector in projectors.items():
            param_count = sum(p.numel() for p in projector.parameters())
            print(f"{name} 投影器:")
            print(f"  参数量: {param_count:,}")
            print(f"  输入维度: {self.clip_hidden_size}")
            print(f"  输出维度: {self.llm_hidden_size}")
        
        return projectors
    
    def process_image_features(self, image_features: torch.Tensor, 
                             projectors: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """处理图像特征并投影"""
        
        results = {}
        
        print("\n=== 图像特征投影处理 ===")
        for proj_name, projector in projectors.items():
            with torch.no_grad():
                projected_features = projector(image_features)
            
            results[proj_name] = projected_features
            
            print(f"\n{proj_name.upper()} 投影结果:")
            print(f"  输出形状: {projected_features.shape}")
            print(f"  输出统计:")
            for i in range(self.batch_size):
                stats = {
                    'mean': projected_features[i].mean().item(),
                    'std': projected_features[i].std().item(),
                    'max': projected_features[i].max().item(),
                    'min': projected_features[i].min().item()
                }
                print(f"    图片{i+1}: {stats}")
            
            # 分析投影效果
            input_norm = image_features.norm(dim=-1).mean().item()
            output_norm = projected_features.norm(dim=-1).mean().item()
            print(f"  特征范数变化: {input_norm:.4f} → {output_norm:.4f}")
        
        return results
    
    def simulate_text_generation(self, projected_features: torch.Tensor, 
                               projector_name: str) -> List[str]:
        """模拟文本生成过程"""
        
        print(f"\n=== {projector_name.upper()} 文本生成模拟 ===")
        
        generated_descriptions = []
        
        for i in range(self.batch_size):
            # 模拟语言模型处理投影后的特征
            img_features = projected_features[i]  # (196, 4096)
            
            # 简化的文本生成模拟
            # 在实际应用中，这里会使用真正的语言模型
            if projector_name == 'linear':
                # 线性投影器生成较简单的描述
                if i == 0:
                    desc = "这是一只猫"
                elif i == 1:
                    desc = "这是一只狗"
                else:
                    desc = "这是一只鸟"
            else:
                # MLP投影器生成更详细的描述
                if i == 0:
                    desc = "一只灰色的猫舒适地坐在沙发上"
                elif i == 1:
                    desc = "一只棕色的狗正在绿色的草地上快乐地奔跑"
                else:
                    desc = "一只色彩斑斓的小鸟稳稳地站在树枝上"
            
            generated_descriptions.append(desc)
            print(f"图片{i+1} 生成描述: {desc}")
            
            # 模拟生成过程的"置信度"
            confidence = 0.7 + 0.25 * (img_features.mean().item() + 1) / 2
            print(f"  生成置信度: {confidence:.3f}")
        
        return generated_descriptions
    
    def run_full_pipeline(self):
        """运行完整的图像描述生成流程"""
        
        print("=" * 60)
        print("图像描述生成任务 - 完整数据流示例")
        print("=" * 60)
        
        # 1. 生成图像特征
        image_features = self.create_sample_image_features()
        
        # 2. 创建投影器
        projectors = self.create_projectors()
        
        # 3. 投影处理
        projected_results = self.process_image_features(image_features, projectors)
        
        # 4. 文本生成模拟
        for proj_name, proj_features in projected_results.items():
            generated_texts = self.simulate_text_generation(proj_features, proj_name)
            
            print(f"\n{proj_name.upper()} 投影器生成结果总结:")
            for i, (original, generated) in enumerate(zip(self.image_descriptions, generated_texts)):
                print(f"  图片{i+1}:")
                print(f"    原始描述: {original}")
                print(f"    生成描述: {generated}")

# 运行示例
if __name__ == "__main__":
    task = ImageDescriptionTask()
    task.run_full_pipeline()
```

### 场景2：视频动作识别任务

#### 任务描述
输入一段视频，识别视频中人的动作。需要将长序列的视频特征压缩并投影到适合分类的特征空间。

#### 数据流示例

```python
class VideoActionRecognitionTask:
    def __init__(self):
        # 视频编码器配置
        self.languagebind_hidden_size = 1024  # LanguageBind特征维度
        self.llm_hidden_size = 4096           # 语言模型特征维度
        self.batch_size = 2                   # 2个视频
        self.frames = 8                        # 每个视频8帧
        self.patches_per_frame = 196           # 每帧196个patches
        
        # Q-Former配置
        self.num_query_tokens = 64            # 查询token数量
        
        # 动作类别
        self.action_categories = ["走路", "挥手", "坐下", "跳跃", "挥手"]
        
    def create_sample_video_features(self) -> torch.Tensor:
        """创建虚构的视频特征数据"""
        
        # 2个视频，每个视频8帧，共1568个patches（8×196）
        video_features = torch.randn(
            self.batch_size, 
            self.frames * self.patches_per_frame, 
            self.languagebind_hidden_size
        )
        
        # 视频1: 走路动作
        # 为走路动作添加时序特征
        for frame_idx in range(self.frames):
            start_idx = frame_idx * self.patches_per_frame
            end_idx = start_idx + self.patches_per_frame
            
            # 模拟走路动作的周期性特征
            phase = 2 * np.pi * frame_idx / self.frames
            walking_pattern = 0.3 * np.sin(phase)
            
            # 在中心区域添加走路特征
            center_start = start_idx + 80
            center_end = start_idx + 120
            video_features[0, center_start:center_end, :200] += walking_pattern
            
            # 添加腿部运动特征
            leg_phase = phase + np.pi/4
            leg_pattern = 0.4 * np.sin(leg_phase)
            video_features[0, center_start:center_end, 200:400] += leg_pattern
        
        # 视频2: 挥手动作
        # 为挥手动作添加时序特征
        for frame_idx in range(self.frames):
            start_idx = frame_idx * self.patches_per_frame
            end_idx = start_idx + self.patches_per_frame
            
            # 模拟挥手动作的周期性特征
            phase = 2 * np.pi * frame_idx / self.frames
            waving_pattern = 0.5 * np.sin(phase * 2)  # 挥手频率更高
            
            # 在上半部分添加手臂运动特征
            upper_start = start_idx + 40
            upper_end = start_idx + 100
            video_features[1, upper_start:upper_end, :300] += waving_pattern
            
            # 添加手部特征
            hand_pattern = 0.6 * np.sin(phase * 2 + np.pi/6)
            video_features[1, upper_start:upper_end, 300:500] += hand_pattern
        
        print("=== 视频特征数据生成 ===")
        print(f"生成视频特征形状: {video_features.shape}")
        print(f"视频动作: ['走路', '挥手']")
        
        for i in range(self.batch_size):
            feat_stats = {
                'mean': video_features[i].mean().item(),
                'std': video_features[i].std().item(),
                'max': video_features[i].max().item(),
                'min': video_features[i].min().item()
            }
            action_name = "走路" if i == 0 else "挥手"
            print(f"视频{i+1} ({action_name}): {feat_stats}")
        
        return video_features
    
    def create_qformer_projector(self) -> nn.Module:
        """创建Q-Former投影器"""
        
        class QFormerProjector(nn.Module):
            def __init__(self, input_dim, output_dim, num_queries):
                super().__init__()
                self.query_tokens = nn.Parameter(torch.randn(1, num_queries, input_dim))
                self.cross_attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
                self.norm1 = nn.LayerNorm(input_dim)
                self.norm2 = nn.LayerNorm(input_dim)
                self.proj = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim)
                )
            
            def forward(self, x):
                batch_size = x.shape[0]
                query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                
                # 交叉注意力
                attended, _ = self.cross_attention(query_tokens, x, x)
                attended = self.norm1(attended + query_tokens)
                
                # 最终投影
                output = self.proj(attended)
                return output
        
        projector = QFormerProjector(
            self.languagebind_hidden_size,
            self.llm_hidden_size,
            self.num_query_tokens
        )
        
        print(f"\n=== Q-Former投影器创建 ===")
        print(f"输入维度: {self.languagebind_hidden_size}")
        print(f"输出维度: {self.llm_hidden_size}")
        print(f"查询token数量: {self.num_query_tokens}")
        print(f"参数量: {sum(p.numel() for p in projector.parameters()):,}")
        
        return projector
    
    def process_video_features(self, video_features: torch.Tensor, 
                             projector: nn.Module) -> torch.Tensor:
        """处理视频特征"""
        
        print("\n=== 视频特征处理 ===")
        print(f"输入视频特征形状: {video_features.shape}")
        print(f"序列长度: {video_features.shape[1]}")
        
        with torch.no_grad():
            projected_features = projector(video_features)
        
        print(f"投影后特征形状: {projected_features.shape}")
        print(f"压缩后序列长度: {projected_features.shape[1]}")
        print(f"压缩比: {projected_features.shape[1] / video_features.shape[1]:.2%}")
        
        # 分析压缩效果
        for i in range(self.batch_size):
            input_norm = video_features[i].norm(dim=-1).mean().item()
            output_norm = projected_features[i].norm(dim=-1).mean().item()
            print(f"视频{i+1}特征范数: {input_norm:.4f} → {output_norm:.4f}")
        
        return projected_features
    
    def action_classification(self, projected_features: torch.Tensor) -> List[str]:
        """模拟动作分类"""
        
        print("\n=== 动作分类模拟 ===")
        
        predicted_actions = []
        
        for i in range(self.batch_size):
            # 使用投影后的特征进行分类
            features = projected_features[i]  # (64, 4096)
            
            # 简化的分类逻辑（实际应用中会使用分类器）
            feature_mean = features.mean().item()
            feature_std = features.std().item()
            
            # 基于特征统计的简单分类规则
            if i == 0:  # 走路视频
                if feature_std > 0.5:
                    predicted_action = "走路"
                    confidence = 0.85
                else:
                    predicted_action = "站立"
                    confidence = 0.6
            else:  # 挥手视频
                if feature_std > 0.7:
                    predicted_action = "挥手"
                    confidence = 0.9
                else:
                    predicted_action = "举手"
                    confidence = 0.65
            
            predicted_actions.append(predicted_action)
            print(f"视频{i+1}:")
            print(f"  预测动作: {predicted_action}")
            print(f"  置信度: {confidence:.3f}")
            print(f"  特征统计: mean={feature_mean:.4f}, std={feature_std:.4f}")
        
        return predicted_actions
    
    def run_full_pipeline(self):
        """运行完整的视频动作识别流程"""
        
        print("=" * 60)
        print("视频动作识别任务 - 完整数据流示例")
        print("=" * 60)
        
        # 1. 生成视频特征
        video_features = self.create_sample_video_features()
        
        # 2. 创建Q-Former投影器
        projector = self.create_qformer_projector()
        
        # 3. 视频特征处理
        projected_features = self.process_video_features(video_features, projector)
        
        # 4. 动作分类
        predicted_actions = self.action_classification(projected_features)
        
        # 5. 结果总结
        print("\n=== 任务完成总结 ===")
        true_actions = ["走路", "挥手"]
        for i, (true, predicted) in enumerate(zip(true_actions, predicted_actions)):
            status = "✓" if true == predicted else "✗"
            print(f"视频{i+1}: {status} 真实:{true} → 预测:{predicted}")

# 运行视频任务示例
if __name__ == "__main__":
    video_task = VideoActionRecognitionTask()
    video_task.run_full_pipeline()
```

### 场景3：多模态问答任务

#### 任务描述
输入图像和问题，生成对应的答案。需要将图像特征和文本特征都投影到统一空间进行融合。

#### 数据流示例

```python
class MultimodalQATask:
    def __init__(self):
        # 多模态配置
        self.batch_size = 2
        self.clip_hidden_size = 768      # 图像特征维度
        self.text_hidden_size = 768      # 文本特征维度
        self.llm_hidden_size = 4096      # 统一特征维度
        
        # QA对示例
        self.qa_pairs = [
            {
                "image": "一张桌子上有一个苹果和一个橙子",
                "question": "桌子上有什么水果？",
                "answer": "苹果和橙子"
            },
            {
                "image": "一个男孩在公园里踢足球",
                "question": "男孩在做什么运动？",
                "answer": "踢足球"
            }
        ]
    
    def create_multimodal_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建多模态特征数据"""
        
        # 图像特征
        image_features = torch.randn(self.batch_size, 196, self.clip_hidden_size)
        
        # 为不同图像添加特定特征
        # 图像1: 桌子上的水果
        image_features[0, 80:100, :150] += 0.4  # 桌子
        image_features[0, 90:95, 200:250] += 0.6  # 苹果
        image_features[0, 95:100, 250:300] += 0.5  # 橙子
        
        # 图像2: 踢足球的男孩
        image_features[1, 70:90, :180] += 0.3  # 男孩
        image_features[1, 100:110, 150:200] += 0.7  # 足球
        image_features[1, 60:80, 180:220] += 0.4  # 公园背景
        
        # 文本特征（问题）
        text_features = torch.randn(self.batch_size, 30, self.text_hidden_size)
        
        # 为问题添加特定特征
        text_features[0, :10, :100] += 0.5  # "桌子上"
        text_features[0, 10:20, 100:200] += 0.6  # "有什么"
        text_features[0, 20:30, 200:300] += 0.4  # "水果"
        
        text_features[1, :8, :120] += 0.4  # "男孩"
        text_features[1, 8:16, 120:220] += 0.5  # "在做什么"
        text_features[1, 16:30, 220:320] += 0.6  # "运动"
        
        print("=== 多模态特征生成 ===")
        print(f"图像特征形状: {image_features.shape}")
        print(f"文本特征形状: {text_features.shape}")
        
        for i, qa in enumerate(self.qa_pairs):
            print(f"QA对{i+1}:")
            print(f"  图像: {qa['image']}")
            print(f"  问题: {qa['question']}")
            print(f"  答案: {qa['answer']}")
        
        return image_features, text_features
    
    def create_multimodal_projectors(self) -> Dict[str, nn.Module]:
        """创建多模态投影器"""
        
        projectors = {}
        
        # 图像投影器
        projectors['image'] = nn.Sequential(
            nn.Linear(self.clip_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        )
        
        # 文本投影器
        projectors['text'] = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        )
        
        print("\n=== 多模态投影器创建 ===")
        for modality, projector in projectors.items():
            param_count = sum(p.numel() for p in projector.parameters())
            print(f"{modality} 投影器:")
            print(f"  参数量: {param_count:,}")
            print(f"  输入维度: {self.clip_hidden_size if modality == 'image' else self.text_hidden_size}")
            print(f"  输出维度: {self.llm_hidden_size}")
        
        return projectors
    
    def process_multimodal_features(self, image_features: torch.Tensor, 
                                   text_features: torch.Tensor,
                                   projectors: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """处理多模态特征"""
        
        print("\n=== 多模态特征处理 ===")
        
        results = {}
        
        # 分别投影图像和文本特征
        with torch.no_grad():
            projected_image = projectors['image'](image_features)
            projected_text = projectors['text'](text_features)
        
        results['image'] = projected_image
        results['text'] = projected_text
        
        print(f"投影后图像特征: {projected_image.shape}")
        print(f"投影后文本特征: {projected_text.shape}")
        
        # 多模态融合（简单拼接）
        fused_features = torch.cat([projected_image, projected_text], dim=1)
        results['fused'] = fused_features
        
        print(f"融合后特征: {fused_features.shape}")
        
        return results
    
    def answer_generation(self, fused_features: torch.Tensor) -> List[str]:
        """模拟答案生成"""
        
        print("\n=== 答案生成模拟 ===")
        
        generated_answers = []
        
        for i in range(self.batch_size):
            # 使用融合特征生成答案
            features = fused_features[i]  # (226, 4096)
            
            # 简化的答案生成逻辑
            feature_mean = features.mean().item()
            feature_std = features.std().item()
            
            if i == 0:  # 水果问题
                if feature_std > 0.3:
                    answer = "苹果和橙子"
                    confidence = 0.88
                else:
                    answer = "水果"
                    confidence = 0.6
            else:  # 运动问题
                if feature_std > 0.4:
                    answer = "踢足球"
                    confidence = 0.92
                else:
                    answer = "运动"
                    confidence = 0.65
            
            generated_answers.append(answer)
            print(f"QA对{i+1}:")
            print(f"  生成答案: {answer}")
            print(f"  置信度: {confidence:.3f}")
            print(f"  特征统计: mean={feature_mean:.4f}, std={feature_std:.4f}")
        
        return generated_answers
    
    def run_full_pipeline(self):
        """运行完整的多模态问答流程"""
        
        print("=" * 60)
        print("多模态问答任务 - 完整数据流示例")
        print("=" * 60)
        
        # 1. 生成多模态特征
        image_features, text_features = self.create_multimodal_features()
        
        # 2. 创建多模态投影器
        projectors = self.create_multimodal_projectors()
        
        # 3. 多模态特征处理
        processed_features = self.process_multimodal_features(
            image_features, text_features, projectors
        )
        
        # 4. 答案生成
        generated_answers = self.answer_generation(processed_features['fused'])
        
        # 5. 结果总结
        print("\n=== 任务完成总结 ===")
        for i, (qa, generated) in enumerate(zip(self.qa_pairs, generated_answers)):
            status = "✓" if qa['answer'] == generated else "✗"
            print(f"QA对{i+1}: {status}")
            print(f"  问题: {qa['question']}")
            print(f"  标准答案: {qa['answer']}")
            print(f"  生成答案: {generated}")

# 运行多模态QA任务示例
if __name__ == "__main__":
    qa_task = MultimodalQATask()
    qa_task.run_full_pipeline()
```

## API使用指南

### 1. 基本API使用

```python
from models.multimodal_projector.builder import build_vision_projector

# 基本使用示例
def basic_usage_example():
    """基本API使用示例"""
    
    # 配置
    class Config:
        def __init__(self):
            self.mm_projector_type = 'mlp2x_gelu'
            self.mm_hidden_size = 768
            self.hidden_size = 4096
    
    config = Config()
    
    # 构建投影器
    projector = build_vision_projector(config)
    
    # 使用投影器
    input_features = torch.randn(2, 196, 768)
    output_features = projector(input_features)
    
    print(f"输入形状: {input_features.shape}")
    print(f"输出形状: {output_features.shape}")
    print(f"投影器类型: {config.mm_projector_type}")

basic_usage_example()
```

### 2. 高级API使用

```python
def advanced_usage_example():
    """高级API使用示例"""
    
    # 不同的投影器配置
    projector_configs = [
        'linear',
        'mlp2x_gelu', 
        'mlp3x_gelu',
        'identity'
    ]
    
    class Config:
        def __init__(self, proj_type):
            self.mm_projector_type = proj_type
            self.mm_hidden_size = 1024
            self.hidden_size = 4096
    
    print("=== 高级API使用示例 ===")
    
    for proj_type in projector_configs:
        try:
            config = Config(proj_type)
            projector = build_vision_projector(config)
            
            # 测试不同输入尺寸
            test_inputs = [
                (2, 196, 1024),   # 标准图像
                (1, 1568, 1024),  # 长序列视频
                (4, 49, 1024),    # 短序列
            ]
            
            print(f"\n{proj_type.upper()} 投影器:")
            print(f"参数量: {sum(p.numel() for p in projector.parameters()):,}")
            
            for input_shape in test_inputs:
                test_input = torch.randn(input_shape)
                with torch.no_grad():
                    output = projector(test_input)
                print(f"  {input_shape} → {output.shape}")
                
        except Exception as e:
            print(f"\n{proj_type.upper()} 投影器创建失败: {e}")

advanced_usage_example()
```

### 3. 实际应用集成

```python
def real_world_integration_example():
    """实际应用集成示例"""
    
    # 模拟完整的多模态处理流程
    class MultimodalPipeline:
        def __init__(self):
            # 编码器配置
            self.vision_encoder_config = type('Config', (), {
                'mm_projector_type': 'mlp2x_gelu',
                'mm_hidden_size': 768,
                'hidden_size': 4096
            })()
            
            # 构建组件
            self.vision_projector = build_vision_projector(self.vision_encoder_config)
            self.language_projector = nn.Linear(768, 4096)
            
        def process_multimodal_input(self, image_features, text_features):
            """处理多模态输入"""
            
            print("=== 多模态处理管道 ===")
            print(f"输入图像特征: {image_features.shape}")
            print(f"输入文本特征: {text_features.shape}")
            
            # 投影处理
            with torch.no_grad():
                projected_image = self.vision_projector(image_features)
                projected_text = self.language_projector(text_features)
                
                # 特征融合
                fused_features = torch.cat([projected_image, projected_text], dim=1)
            
            print(f"投影后图像: {projected_image.shape}")
            print(f"投影后文本: {projected_text.shape}")
            print(f"融合后特征: {fused_features.shape}")
            
            return fused_features
        
        def generate_response(self, fused_features):
            """生成响应"""
            
            # 简化的响应生成
            batch_size = fused_features.shape[0]
            responses = []
            
            for i in range(batch_size):
                features = fused_features[i]
                
                # 基于特征统计生成不同响应
                feature_norm = features.norm().item()
                
                if feature_norm > 100:
                    response = "这是一个复杂的多模态场景"
                elif feature_norm > 50:
                    response = "这是一个中等复杂度的场景"
                else:
                    response = "这是一个简单的场景"
                
                responses.append(response)
                print(f"样本{i+1}: {response} (特征范数: {feature_norm:.2f})")
            
            return responses
    
    # 运行完整示例
    pipeline = MultimodalPipeline()
    
    # 模拟输入数据
    image_features = torch.randn(2, 196, 768)
    text_features = torch.randn(2, 30, 768)
    
    # 处理流程
    fused_features = pipeline.process_multimodal_input(image_features, text_features)
    responses = pipeline.generate_response(fused_features)
    
    print("\n=== 集成示例完成 ===")

real_world_integration_example()
```

## 最佳实践和调试指南

### 1. 投影器选择指南

```python
def projector_selection_guide():
    """投影器选择指南"""
    
    print("=== 投影器选择指南 ===")
    
    scenarios = [
        {
            "name": "计算资源受限",
            "recommendation": "linear",
            "reason": "参数量最少，计算效率最高"
        },
        {
            "name": "特征维度差异大",
            "recommendation": "mlp2x_gelu",
            "reason": "非线性变换能力更强"
        },
        {
            "name": "长序列处理",
            "recommendation": "qformer2_64",
            "reason": "可以压缩序列长度"
        },
        {
            "name": "调试和开发",
            "recommendation": "identity",
            "reason": "保持原始特征，便于调试"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print(f"推荐投影器: {scenario['recommendation']}")
        print(f"原因: {scenario['reason']}")

projector_selection_guide()
```

### 2. 性能监控和调试

```python
def performance_monitoring_example():
    """性能监控和调试示例"""
    
    class PerformanceMonitor:
        def __init__(self, projector):
            self.projector = projector
            self.stats = {}
        
        def benchmark(self, input_shapes, num_runs=100):
            """性能基准测试"""
            
            import time
            
            print("=== 性能监控 ===")
            
            for shape in input_shapes:
                print(f"\n测试输入形状: {shape}")
                
                # 预热
                dummy_input = torch.randn(shape)
                with torch.no_grad():
                    _ = self.projector(dummy_input)
                
                # 基准测试
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    with torch.no_grad():
                        output = self.projector(dummy_input)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                print(f"平均推理时间: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
                print(f"吞吐量: {1/avg_time:.2f} samples/s")
                
                # 内存使用
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    dummy_input = dummy_input.cuda()
                    self.projector = self.projector.cuda()
                    
                    with torch.no_grad():
                        _ = self.projector(dummy_input)
                    
                    memory_used = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"GPU内存使用: {memory_used:.2f} MB")
    
    # 创建监控器并测试
    config = type('Config', (), {
        'mm_projector_type': 'mlp2x_gelu',
        'mm_hidden_size': 768,
        'hidden_size': 4096
    })()
    
    projector = build_vision_projector(config)
    monitor = PerformanceMonitor(projector)
    
    test_shapes = [
        (1, 196, 768),    # 单样本
        (4, 196, 768),    # 小批量
        (8, 196, 768),    # 中批量
        (16, 196, 768),   # 大批量
    ]
    
    monitor.benchmark(test_shapes)

performance_monitoring_example()
```

通过这些详细的API使用指南和数据流示例，您可以全面了解MotionLLM多模态投影器的实际应用方式，并在自己的项目中灵活运用。