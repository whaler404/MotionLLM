# 模型结构与张量形状

## 多模态主干 `LlavaMetaModel`
`CLI.py` 中的 `LlavaMetaModel` 将 LanguageBind 感知塔与线性投影器组合起来，为语言模型提供视觉前缀特征。【F:CLI.py†L34-L145】

### 视频分支前向流程
1. 视频预处理后，处理器返回形状为 `(B, C, T, H, W)` 的张量。例如交互式推理阶段读取单个视频时，张量为 `(1, 3, 8, 224, 224)`。【F:CLI.py†L289-L297】
2. `LanguageBindVideoTower` 将输入映射为 `(B, N, 1024)` 的 token 序列（代码中注释列出了 `torch.Size([1, 2048, 1024])`）。【F:CLI.py†L135-L139】
3. `mm_projector` 再把视觉 token 投射到语言模型隐藏维度 `(B, N, 4096)`，为后续与文本 token 融合做好准备。【F:CLI.py†L135-L139】
4. `get_multimodal_embeddings` 根据模态关键字动态调用 `encode_videos` 或 `encode_images`，统一输出视觉特征序列。【F:CLI.py†L141-L145】

### 感知塔构建
`models/multimodal_encoder/builder.py` 根据配置名称选择具体骨干：
- 图像塔根据路径前缀在 CLIP、LanguageBind 或 MAE 实现之间切换。【F:models/multimodal_encoder/builder.py†L7-L21】
- 视频塔默认使用 `LanguageBind_Video_merge`，其输出形状为 `(batch, frames * patches, hidden)`，随后由多模态投影器完成维度匹配。【F:models/multimodal_encoder/builder.py†L23-L30】

## 文本生成模型
- `GPT` 来自 `lit_gpt.lora`，在加载基础 Vicuna 权重后，附加 LoRA 模块以降低微调成本。【F:CLI.py†L8-L27】【F:CLI.py†L222-L245】
- 视觉投影器的权重与 LoRA 权重一起合并进语言模型，使得 `(N, 4096)` 的视觉嵌入可以与 tokenizer 输出的 `(T, 4096)` 文本隐藏状态串接。【F:CLI.py†L260-L313】

## 关键形状总结
| 阶段 | 形状 | 说明 |
| --- | --- | --- |
| 视频处理器输出 | `(B, C, T, H, W)` | 通过 `video_processor` 完成归一化与采样。【F:CLI.py†L289-L297】 |
| 视频塔输出 | `(B, N, 1024)` | LanguageBind 视频编码器提取的时空特征。【F:CLI.py†L135-L139】 |
| 多模态投影输出 | `(B, N, 4096)` | 对齐 Vicuna 隐藏维度，便于拼接文本 tokens。【F:CLI.py†L135-L139】 |
| 文本 token 序列 | `(1, T)` | 由 tokenizer 编码 prompt 与回答。【F:CLI.py†L308-L327】【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L85-L111】 |

视觉序列最终通过 `generate` 函数与文本 tokens 一同送入语言模型，完成跨模态的回答生成。【F:CLI.py†L317-L331】
