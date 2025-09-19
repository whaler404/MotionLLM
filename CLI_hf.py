import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None
from scripts.video_dataset.prepare_video_dataset_video_llava import generate_prompt_mlp
from options import option
from models.multimodal_encoder.builder import build_image_tower, build_video_tower
from models.multimodal_projector.builder import build_vision_projector


warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CLI_HF")

args = option.get_args_parser()


def _log_arguments(arguments):
    key_color = "\033[1;36m"
    value_color = "\033[1;33m"
    reset = "\033[0m"
    logger.info("CLI arguments:")
    for key in sorted(vars(arguments)):
        value = getattr(arguments, key)
        logger.info("  %s%-24s%s : %s%s%s", key_color, key, reset, value_color, value, reset)


_log_arguments(args)


def _normalize_device_string(device_value):
    if device_value is None:
        return None
    value = device_value.strip()
    if not value:
        return None
    if value.isdigit():
        return f"cuda:{value}"
    if value.startswith("cuda") and ":" not in value and value != "cuda":
        suffix = value[len("cuda"):]
        if suffix.isdigit():
            return f"cuda:{suffix}"
    return value


class LlavaMetaModel:
    def __init__(self, config, pretrained_checkpoint):
        super().__init__()
        self.mm_projector_device = None
        self.mm_projector_dtype = None
        if hasattr(config, "mm_image_tower") or hasattr(config, "image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower") or hasattr(config, "video_tower"):
            self.video_tower = build_video_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.load_video_tower_pretrained(pretrained_checkpoint)

    def get_image_tower(self):
        image_tower = getattr(self, "image_tower", None)
        if isinstance(image_tower, list):
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, "video_tower", None)
        if isinstance(video_tower, list):
            video_tower = video_tower[0]
        return video_tower

    def get_all_tower(self, keys):
        return {key: getattr(self, f"get_{key}_tower") for key in keys}

    def load_video_tower_pretrained(self, pretrained_checkpoint):
        self.mm_projector.load_state_dict(pretrained_checkpoint, strict=True)

    def configure_projector(self, device=None, dtype=None):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if kwargs:
            self.mm_projector = self.mm_projector.to(**kwargs)
        sample_param = next(self.mm_projector.parameters())
        self.mm_projector_device = sample_param.device
        self.mm_projector_dtype = sample_param.dtype

    def encode_images(self, images):
        image_features = self.get_image_tower()(images)
        if self.mm_projector_device is not None and image_features.device != self.mm_projector_device:
            image_features = image_features.to(self.mm_projector_device)
        if self.mm_projector_dtype is not None and image_features.dtype != self.mm_projector_dtype:
            image_features = image_features.to(dtype=self.mm_projector_dtype)
        image_features = self.mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        video_features = self.get_video_tower()(videos)
        if self.mm_projector_device is not None and video_features.device != self.mm_projector_device:
            video_features = video_features.to(self.mm_projector_device)
        if self.mm_projector_dtype is not None and video_features.dtype != self.mm_projector_dtype:
            video_features = video_features.to(dtype=self.mm_projector_dtype)
        video_features = self.mm_projector(video_features)
        return video_features

    def get_multimodal_embeddings(self, X_modalities):
        Xs, keys = X_modalities
        return getattr(self, f"encode_{keys[0]}s")(Xs)


def get_processor(
    X,
    config,
    devices,
    pretrained_checkpoint_tower,
    dtype,
    model_path="LanguageBind/Video-LLaVA-7B",
):
    processor = {}
    mm_backbone_mlp_model = LlavaMetaModel(config, pretrained_checkpoint_tower)
    projector_device = devices.get("projector")
    mm_backbone_mlp_model.configure_projector(device=projector_device, dtype=dtype)

    if "Image" in X:
        image_tower = mm_backbone_mlp_model.get_image_tower()
        if not image_tower.is_loaded:
            logger.info("Loading image tower weights to %s", devices.get("image"))
            image_tower.load_model()
        image_device = devices.get("image")
        image_tower.to(device=image_device, dtype=dtype)
        processor["image"] = image_tower.image_processor

    if "Video" in X:
        video_tower = mm_backbone_mlp_model.get_video_tower()
        if not video_tower.is_loaded:
            logger.info("Loading video tower weights to %s", devices.get("video"))
            video_tower.load_model()
        video_device = devices.get("video")
        video_tower.to(device=video_device, dtype=dtype)
        processor["video"] = video_tower.video_processor

    return mm_backbone_mlp_model, processor


def _resolve_device(candidate, fallback):
    normalized = _normalize_device_string(candidate)
    if normalized is None:
        return fallback
    return torch.device(normalized)


def _load_tokenizer_and_model(checkpoint_dir, dtype, llm_device, lora_path=None):
    logger.info("Loading tokenizer from %s", checkpoint_dir)
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading base Vicuna model to %s", llm_device)
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.to(llm_device)
    model.eval()

    if lora_path:
        logger.info("Applying LoRA adapter from %s", lora_path)
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters in CLI_hf")
        if os.path.isdir(lora_path):
            model = PeftModel.from_pretrained(model, lora_path, torch_dtype=dtype)
            model.to(llm_device)
        else:
            raise ValueError(
                f"LoRA path {lora_path} is not a PEFT directory. Convert the adapter to Hugging Face format before use."
            )

    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def _build_inputs(tokenizer, model, prefix_text, suffix_text, video_feature, llm_device):
    prefix_ids = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(llm_device)
    input_video_tag_ids = tokenizer(
        "INPUT_VIDEO: ",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(llm_device)
    prefix_ids = torch.cat([prefix_ids, input_video_tag_ids], dim=1)

    suffix_ids = tokenizer(
        suffix_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(llm_device)

    embed_layer = model.get_input_embeddings()
    prefix_embeds = embed_layer(prefix_ids)
    suffix_embeds = embed_layer(suffix_ids)
    video_embeds = video_feature.to(llm_device)

    inputs_embeds = torch.cat([prefix_embeds, video_embeds, suffix_embeds], dim=1)
    placeholder_ids = torch.full(
        (video_embeds.size(0), video_embeds.size(1)),
        tokenizer.pad_token_id,
        device=llm_device,
        dtype=prefix_ids.dtype,
    )
    input_ids = torch.cat([prefix_ids, placeholder_ids, suffix_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    return input_ids, inputs_embeds, attention_mask, prefix_ids.size(1) + video_embeds.size(1) + suffix_ids.size(1)


def _generate_with_video(
    model,
    tokenizer,
    input_ids,
    inputs_embeds,
    attention_mask,
    prompt_length,
    max_new_tokens,
    temperature,
    top_k,
):
    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature is not None and temperature > 0,
        "temperature": temperature,
        "top_k": top_k,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    outputs = model.generate(**generation_kwargs)
    generated = outputs[:, prompt_length:]
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main(
    quantize: Optional[str] = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",
) -> None:
    if quantize is not None:
        raise NotImplementedError("Quantization is not supported in CLI_hf yet")

    device_overrides = args.device_map or {}
    llm_device_str = _normalize_device_string(device_overrides.get("llm", args.llm_device))
    if llm_device_str is None:
        raise ValueError("llm device must be specified either via --llm_device or --device_map.")
    llm_device = torch.device(llm_device_str)

    device_map = {
        "llm": llm_device,
        "image": _resolve_device(device_overrides.get("image", args.image_tower_device), llm_device),
        "video": _resolve_device(device_overrides.get("video", args.video_tower_device), llm_device),
        "projector": _resolve_device(device_overrides.get("projector", args.projector_device), llm_device),
    }
    logger.info("Resolved device map: %s", {k: str(v) for k, v in device_map.items()})

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    checkpoint_dir = 'lmsys/vicuna-7b-v1.5'
    # if not checkpoint_dir.exists():
    #     raise FileNotFoundError(f"Vicuna checkpoint not found at {checkpoint_dir}")

    logger.info("Preparing Vicuna backbone and tokenizer")
    tokenizer, model = _load_tokenizer_and_model(
        checkpoint_dir,
        dtype,
        llm_device,
        lora_path=args.lora_path,
    )

    accelerator = accelerator if accelerator != "auto" else ("cuda" if llm_device.type == "cuda" else llm_device.type)
    fabric_devices = 1
    if llm_device.type == "cuda" and llm_device.index is not None:
        fabric_devices = [llm_device.index]
        accelerator = "cuda"
    fabric = L.Fabric(accelerator=accelerator, devices=fabric_devices)

    mlp_path = args.mlp_path
    logger.info("Loading projector checkpoint from %s", mlp_path)
    pretrained_checkpoint_mlp = torch.load(mlp_path)

    X = ["Video"]
    logger.info("Initializing multimodal towers and processors")
    mm_backbone_mlp_model, processor = get_processor(
        X,
        args,
        device_map,
        pretrained_checkpoint_mlp,
        dtype,
        model_path="LanguageBind/Video-LLaVA-7B",
    )
    video_processor = processor["video"]
    linear_proj = mm_backbone_mlp_model.mm_projector
    linear_proj.load_state_dict(pretrained_checkpoint_mlp)
    linear_proj = linear_proj.to(device=device_map["projector"], dtype=dtype)
    linear_proj.eval()

    logger.info("Vicuna + tokenizer ready; LoRA adapter: %s", args.lora_path if args.lora_path else "(none)")

    model.eval()
    model = fabric.setup_module(model)

    video_device = device_map["video"]

    while True:
        logger.info("Awaiting user video input")
        input_video_path = input("\033[0;34;40m Input video path: \033[0m")
        video_tensor = video_processor(input_video_path, return_tensors="pt")["pixel_values"]

        if isinstance(video_tensor, list):
            tensor = [video.to(video_device, dtype=dtype) for video in video_tensor]
        else:
            tensor = video_tensor.to(video_device, dtype=dtype)

        logger.info("Encoding video features")
        X_modalities = [tensor, ["video"]]
        video_feature = mm_backbone_mlp_model.get_multimodal_embeddings(X_modalities)
        video_feature = video_feature.to(llm_device, dtype=dtype)

        logger.info("Awaiting user prompt")
        prompt = input("\033[0;34;40m Your question: \033[0m")
        sample = {"instruction": prompt, "input": input_video_path}
        prefix_template = generate_prompt_mlp(sample)
        prefix_text = prefix_template.split("INPUT_VIDEO: ")[0] + "\n"
        suffix_text = ". ASSISTANT: "

        input_ids, inputs_embeds, attention_mask, prompt_length = _build_inputs(
            tokenizer,
            model,
            prefix_text,
            suffix_text,
            video_feature,
            llm_device,
        )

        attention_mask = attention_mask.to(llm_device)
        input_ids = input_ids.to(llm_device)
        inputs_embeds = inputs_embeds.to(llm_device, dtype=dtype)

        logger.info("Generating response (max_new_tokens=%d)", max_new_tokens)
        output = _generate_with_video(
            model,
            tokenizer,
            input_ids,
            inputs_embeds,
            attention_mask,
            prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        logger.info("Generation finished")
        print("================================")
        print("Model output", output.strip())
        print("================================")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
