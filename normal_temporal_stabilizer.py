
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LOG = logging.getLogger("VizClickNormalStabilizer")


def _to_uint8_rgb(frame):
    """ComfyUI IMAGE [H,W,C] float RGB -> uint8 RGB."""
    x = frame.detach().float().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _normal_to_vectors(frame):
    """[H,W,3] normal visualization [0,1] -> unit vectors [-1,1]."""
    n = frame.float() * 2.0 - 1.0
    n = F.normalize(n, dim=-1, eps=1e-6)
    return n


def _vectors_to_normal(n):
    n = F.normalize(n, dim=-1, eps=1e-6)
    return (n + 1.0) * 0.5


def _warp_rgb(prev, flow):
    """
    Warp previous normal frame into current coordinates.
    Farneback flow is prev->current, so sample prev at current-flow.
    prev: [H,W,3] float32
    flow: [H,W,2] float32
    """
    h, w = prev.shape[:2]
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = gx - flow[..., 0]
    map_y = gy - flow[..., 1]
    out = cv2.remap(
        prev,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return out


class VizClickNormalTemporalStabilizer:
    """
    Motion-aware temporal stabilizer for video normal maps.

    The source video is used only to estimate optical flow. The normal map
    itself is never motion-blurred: the previous stabilized normal is warped
    into the current frame, then blended adaptively with the current normal
    according to temporal consistency.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE",),
                "normal_video": ("IMAGE",),
                "temporal_strength": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "motion_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "consistency_threshold": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stabilized_normals",)
    FUNCTION = "stabilize"
    CATEGORY = "VizClick/Video"

    def stabilize(
        self,
        source_images,
        normal_video,
        temporal_strength=0.65,
        motion_strength=1.0,
        consistency_threshold=0.12,
    ):
        if source_images.ndim != 4 or normal_video.ndim != 4:
            raise RuntimeError("Expected IMAGE batches for source_images and normal_video.")

        n = min(source_images.shape[0], normal_video.shape[0])
        if n < 1:
            raise RuntimeError("The input video contains no frames.")

        # Work at normal-map resolution. Source frames are resized only for
        # optical-flow estimation; the normal map keeps its original output size.
        oh, ow = normal_video.shape[1], normal_video.shape[2]

        source = source_images[:n].float()
        if source.shape[1] != oh or source.shape[2] != ow:
            source = source.permute(0, 3, 1, 2)
            source = F.interpolate(
                source, size=(oh, ow), mode="bilinear", align_corners=False
            )
            source = source.permute(0, 2, 3, 1).contiguous()

        normals = normal_video[:n].float().clamp(0, 1)

        result = [normals[0].cpu()]
        prev_source = _to_uint8_rgb(source[0])
        prev_stable = _normal_to_vectors(normals[0].cpu())

        # Farneback settings favor smooth, large-scale motion suitable for
        # stabilizing normals without trying to track individual texture pixels.
        for i in range(1, n):
            cur_source = _to_uint8_rgb(source[i])

            prev_gray = cv2.cvtColor(prev_source, cv2.COLOR_RGB2GRAY)
            cur_gray = cv2.cvtColor(cur_source, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                cur_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
            flow *= float(motion_strength)

            warped_prev = _warp_rgb(
                prev_stable.numpy().astype(np.float32),
                flow,
            )
            warped_prev_t = torch.from_numpy(warped_prev)

            cur_normal = _normal_to_vectors(normals[i].cpu())

            # Angular disagreement in vector space. Small disagreement means
            # the previous normal is a reliable temporal guide.
            dot = (cur_normal * warped_prev_t).sum(dim=-1).clamp(-1.0, 1.0)
            angular_error = torch.acos(dot) / np.pi

            # Adaptive temporal weight:
            # - strong when the warped history agrees with current normals
            # - weak when there is genuine geometry/change or occlusion
            consistency = torch.exp(
                -((angular_error / max(float(consistency_threshold), 1e-4)) ** 2)
            )
            alpha = float(temporal_strength) * consistency.unsqueeze(-1)

            stable = alpha * warped_prev_t + (1.0 - alpha) * cur_normal
            stable = F.normalize(stable, dim=-1, eps=1e-6)

            result.append(_vectors_to_normal(stable).cpu())
            prev_stable = stable
            prev_source = cur_source

        output = torch.stack(result, dim=0).clamp(0, 1)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "VizClickNormalTemporalStabilizer": VizClickNormalTemporalStabilizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VizClickNormalTemporalStabilizer": "VizClick Normal Temporal Stabilizer"
}
