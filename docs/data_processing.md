# 数据处理流程说明

## 总览
MotionLLM 的视频指令数据由 `scripts/video_dataset/prepare_video_dataset_intern_video.py` 与 `prepare_video_dataset_video_llava.py` 负责转换为模型可直接使用的 PyTorch 序列文件。流程包括：

1. 读取原始 JSON 指令、视频路径与回答文本。`json.load` 的结果会被立即转成列表，以保证后续长度统计和迭代顺序的一致性。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L43-L47】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L43-L47】
2. 遍历样本并构造统一的 prompt，拼接回答后送入分词器得到 `input_ids` 和 `labels`。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L85-L111】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L85-L111】
3. 可选地将 prompt 对应的标签位置填入 `IGNORE_INDEX=-1`，使得训练只在回答部分计算损失。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L102-L107】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L102-L107】
4. 将处理后的样本列表保存为 `.pt` 文件，供后续训练/推理直接加载。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L58-L61】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L58-L61】

## 输入输出格式示例
原始 JSON 中的单个样本大致形如：

```json
{
  "instruction": "Describe the motion in the clip",
  "input": "video_samples/sample_0001.mp4",
  "output": "The person waves and then sits down."
}
```

经过 `prepare_sample` 后会得到包含以下关键字段的字典：

```python
{
  "instruction": "Describe the motion in the clip",
  "input": "video_samples/sample_0001.mp4",
  "output": "The person waves and then sits down.",
  "sys_command": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
  "input_ids_no_response": tensor([1, 892, 298, ...]),
  "input_ids": tensor([1, 892, 298, ..., 29871, 13]),
  "labels": tensor([-1, -1, -1, ..., 29871, 13])
}
```
其中：
- `input_ids_no_response` 仅包含用户指令部分，构造方式见 `tokenize(tokenizer, full_prompt, eos=False)`。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L90-L91】
- `input_ids` 在末尾追加了回答文本，并在调用 `tokenize(..., eos=True)` 时自动附加 `<eos>` 标记。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L85-L92】
- 当 `mask_inputs=True` 时，`labels` 前半段会被填为 `-1` 以忽略用户 prompt。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L102-L107】

## Prompt 模板
两个脚本都复用了 `generate_prompt_mlp`，当样本包含 `input`（视频路径或描述）时，会生成类似：

```
A chat between a curious user and an artificial intelligence assistant, paired with an input that provides further context. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe the motion in the clip INPUT_VIDEO: video_samples/sample_0001.mp4.
ASSISTANT:
```
【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L124-L138】

若无输入，则省略 `INPUT_VIDEO` 字段，直接接在 `ASSISTANT:` 后等待模型生成。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L136-L138】

## 数据质量控制
`prepare` 会针对不同的 `split`（例如 `train_intern_human_2M_stage1_caption` 或 `video_llava_train`）生成对应的 `.pt` 文件，方便后续在不同子任务之间切换。【F:scripts/video_dataset/prepare_video_dataset_intern_video.py†L152-L157】【F:scripts/video_dataset/prepare_video_dataset_video_llava.py†L175-L180】
