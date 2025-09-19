# 多 GPU 推理与设备映射说明

本节记录了 MotionLLM 在同时加载 Vicuna 7B、LLaVA 视频塔、图像塔以及多模态投影器时的 GPU 分配策略，并解释了不同模块之间如何在多卡场景下传递特征。内容基于 `CLI.py` 与 `app.py` 的最新实现。

## 1. 模块拆分概览

| 模块 | 默认设备键 | 主要职责 | 关键入口 |
|------|------------|----------|----------|
| Vicuna 7B (LLM) | `llm` | 负责文本生成，包含 LoRA 参数 | `CLI.py:255-276`, `app.py:554-575` |
| LanguageBind Image Tower | `image` | 静态图像编码，返回 patch 级特征 | `CLI.py:196-205`, `app.py:292-301` |
| LanguageBind Video Tower | `video` | 视频编码，生成时序特征 | `CLI.py:205-214`, `app.py:301-314` |
| Multimodal Projector | `projector` | 将视觉特征映射到 LLM 宽度 | `CLI.py:316-318`, `app.py:636-638` |

所有组件默认继承 `--llm_device`，但现在可以通过新的 `--device_map` 参数为它们指定独立 GPU。

## 2. CLI 参数与设备解析

- `--device_map`: 逗号分隔的 `组件=设备` 列表，例如 `--device_map "llm=cuda:0,video=cuda:1,projector=2"`。纯数字会自动扩展为 `cuda:<id>`。
- 兼容别名：`image_tower`、`llava_image` 会归并到 `image`，`vision_projector`、`mm_projector` 归并到 `projector`，避免记忆不同命名。
- CLI / Gradio 共用解析逻辑：
  ```python
  device_overrides = args.device_map or {}
  llm_device = torch.device(_normalize_device_string(device_overrides.get('llm', args.llm_device)))
  device_map = {
      'image': _resolve_device(device_overrides.get('image', args.image_tower_device), llm_device),
      'video': _resolve_device(device_overrides.get('video', args.video_tower_device), llm_device),
      'projector': _resolve_device(device_overrides.get('projector', args.projector_device), llm_device),
  }
  ```
  因此未显式指定的模块会自动回落到语言模型所在设备。

## 3. 加载顺序与驻留设备

1. **语言模型**：在 `EmptyInitOnDevice` 上下文中于 `llm` 设备初始化，再合并 LoRA 权重。
2. **投影器**：加载检查点后调用 `linear_proj.to(device_map['projector'], dtype)`，确保 MLP 驻留在目标 GPU。
3. **视觉塔**：首次访问 `get_image_tower()` 或 `get_video_tower()` 时，如果尚未加载，会下载权重并立即 `to(device=device_map['image'|'video'])`。
4. **处理器**：每个塔暴露 `image_processor` / `video_processor`，其内部无需设备迁移，仅负责 CPU 上的预处理。

该流程在 CLI 和 Gradio 模式中完全对齐，便于使用相同的参数在两种入口下运行。

## 4. 跨设备数据流

- 视觉塔完成前向后，其输出特征依然保存在塔所在的 GPU。
- 构造 `get_multimodal_embeddings()` 结果后，推理代码显式调用：
  ```python
  video_feature = video_feature.to(llm_device, dtype=dtype)
  ```
  因此即使视频塔部署在 `cuda:1`，也会在发送给 Vicuna 之前迁移至 `llm` 设备 `cuda:0`。
- MLP 投影器由 `configure_projector(device=device_map['projector'], dtype=dtype)` 绑定在目标卡上，塔输出会先迁移到 projector，再投影到 4096 维。
- 文本 token 始终在 `llm` 设备上生成，与视觉特征拼接后进入语言模型。

通过以上迁移步骤，组件可以独立分布在多卡，仍保持正确的数据依赖顺序。

## 5. 实际配置示例

```bash
# 将语言模型放在 cuda:0，视频编码器放在 cuda:1，投影器放在 cuda:2
python CLI.py \
  --lora_path ./checkpoints/motion_llava/lora.pth \
  --mlp_path ./checkpoints/motion_llava/projector.pth \
  --device_map "llm=cuda:0,video=cuda:1,projector=cuda:2"

# Hugging Face 版本 CLI 的等价命令
python CLI_hf.py  --mlp_path ./checkpoints/motion_llava/projector.pth  --device_map "llm=0,video=1,projector=2"
  --lora_path ./hf_adapters/video_qa_adapter \
```
- 若需要同时启用图像塔，可在映射中加入 `image=cuda:3`。
- Gradio Demo 同样支持该参数：`python app.py --device_map "llm=0,video=1,projector=2"`。

运行时会打印解析后的映射，例如：
```
Resolved device map: {'llm': 'cuda:0', 'image': 'cuda:0', 'video': 'cuda:1', 'projector': 'cuda:2'}
```
用于核对是否符合预期。

## 6. 常见注意事项

- 所有设备字符串都会被 `_normalize_device_string` 处理，避免 `cuda0`、`0`、`CUDA:1` 等写法导致解析失败。
- 请确保各 GPU 之间支持 P2P 复制（NVLink/PCIe），否则跨卡张量迁移可能变慢。
- 若 `--device_map` 中遗漏 `llm`，程序会报错提示必须显式指定或使用 `--llm_device`。
- 建议结合 `nvidia-smi` 观察显存占用，确认多模型拆分后每张卡负载符合预期。

通过以上机制，可以在 6x3090 (24GB) 环境中将 Vicuna、LLaVA 视频塔、投影器等模块拆分到不同 GPU，缓解单卡显存压力并提升运行稳定性。
