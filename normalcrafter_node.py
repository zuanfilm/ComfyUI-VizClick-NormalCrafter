import gc
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths
import comfy.model_management

from .bootstrap import ensure_upstream_code

LOG = logging.getLogger("VizClickNormalCrafter")
def _find_model_dir():
    models = Path(folder_paths.models_dir)
    for name in ("NormalCrafter", "Normalcrafter", "normalcrafter"):
        p = models / name
        if p.is_dir():
            return p
    return models / "NormalCrafter"

MODEL_DIR = _find_model_dir()
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
UPSTREAM_DIR = VENDOR_DIR / "normalcrafter"

_PIPE = None
_PIPE_DTYPE = None

def _check_model():
    required = [
        MODEL_DIR / "model_index.json",
        MODEL_DIR / "feature_extractor" / "preprocessor_config.json",
        MODEL_DIR / "image_encoder" / "config.json",
        MODEL_DIR / "image_encoder" / "model.fp16.safetensors",
        MODEL_DIR / "scheduler" / "scheduler_config.json",
        MODEL_DIR / "unet" / "config.json",
        MODEL_DIR / "vae" / "config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Incomplete NormalCrafter model at {MODEL_DIR}. Missing: " + ", ".join(missing)
        )

def _load_pipeline():
    global _PIPE, _PIPE_DTYPE
    _check_model()
    ensure_upstream_code(UPSTREAM_DIR)
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))

    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
    from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter

    device = comfy.model_management.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if _PIPE is not None and _PIPE_DTYPE == dtype:
        return _PIPE

    unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
        str(MODEL_DIR), subfolder="unet", low_cpu_mem_usage=True, torch_dtype=dtype
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(MODEL_DIR), subfolder="vae", low_cpu_mem_usage=True, torch_dtype=dtype
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        str(MODEL_DIR), subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(MODEL_DIR),
        subfolder="image_encoder",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        str(MODEL_DIR), subfolder="scheduler"
    )

    pipe = NormalCrafterPipeline(
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
    )
    pipe = pipe.to(dtype=dtype)

    if device.type == "cuda":
        try:
            pipe.enable_model_cpu_offload(device=device)
        except TypeError:
            pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _PIPE, _PIPE_DTYPE = pipe, dtype
    return pipe

def _safe_resize(images, max_res):
    h, w = images.shape[1:3]
    scale = min(1.0, float(max_res) / max(h, w))
    h2 = max(64, int(round(h * scale)))
    w2 = max(64, int(round(w * scale)))
    h2 = max(64, ((h2 + 63) // 64) * 64)
    w2 = max(64, ((w2 + 63) // 64) * 64)
    x = images.permute(0, 3, 1, 2)
    if (h2, w2) != (h, w):
        x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous(), (h, w)

class VizClickNormalCrafter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "max_res_dimension": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            "window_size": ("INT", {"default": 14, "min": 2, "max": 64}),
            "time_step_size": ("INT", {"default": 10, "min": 1, "max": 64}),
            "decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            "offload_after": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_video",)
    FUNCTION = "process"
    CATEGORY = "VizClick/Video/Geometry"

    def process(self, images, seed, max_res_dimension, window_size,
                time_step_size, decode_chunk_size, offload_after):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE [frames,H,W,3], got {tuple(images.shape)}")
        n = int(images.shape[0])
        if n == 0:
            return (images,)

        pipe = _load_pipeline()
        resized, original_size = _safe_resize(images.float(), max_res_dimension)
        frames = [
            Image.fromarray((frame.clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))
            for frame in resized
        ]
        generator = torch.Generator(device=pipe._execution_device).manual_seed(int(seed))

        with torch.inference_mode():
            result = pipe(
                frames,
                decode_chunk_size=decode_chunk_size,
                time_step_size=time_step_size,
                window_size=window_size,
                generator=generator,
            )

        # Diffusers/NormalCrafter returns video frames as [frames, H, W, C]
        # (or a list of PIL/numpy frames). Convert explicitly to ComfyUI IMAGE
        # layout [frames, H, W, C] before resizing. The previous implementation
        # treated HWC output as NCHW, which caused an enormous CPU allocation.
        output = result.frames[0]

        if isinstance(output, list):
            frames = []
            for frame in output:
                if isinstance(frame, Image.Image):
                    frame = torch.from_numpy(__import__("numpy").array(frame))
                elif not torch.is_tensor(frame):
                    frame = torch.from_numpy(__import__("numpy").asarray(frame))
                frames.append(frame)
            output = torch.stack(frames, dim=0)

        elif isinstance(output, Image.Image):
            output = torch.from_numpy(__import__("numpy").array(output)).unsqueeze(0)

        elif not torch.is_tensor(output):
            output = torch.from_numpy(__import__("numpy").asarray(output))

        if output.ndim != 4:
            raise RuntimeError(
                f"Unexpected NormalCrafter output shape: {tuple(output.shape)}"
            )

        # Normalize possible channel-first output to [T,H,W,C].
        if output.shape[-1] not in (1, 3, 4) and output.shape[1] in (1, 3, 4):
            output = output.permute(0, 2, 3, 1)

        output = output.float()

        # NormalCrafter normally returns normals in [-1, 1]. If the decoder
        # already returned [0, 1], keep that range.
        if output.min() < 0:
            output = (output.clamp(-1, 1) + 1.0) * 0.5
        else:
            output = output.clamp(0, 1)

        oh, ow = original_size
        if output.shape[1] != oh or output.shape[2] != ow:
            x = output.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)
            output = x.permute(0, 2, 3, 1)

        output = output.contiguous().float().cpu()[:n]

        if offload_after:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
        return (output,)

NODE_CLASS_MAPPINGS = {"VizClickNormalCrafter": VizClickNormalCrafter}
NODE_DISPLAY_NAME_MAPPINGS = {"VizClickNormalCrafter": "VizClick NormalCrafter"}
