"""
VizClick NormalCrafter
A minimal ComfyUI integration for the NormalCrafter video-normal model.

The model/inference implementation is intentionally kept separate from the
ComfyUI adapter. Put the upstream NormalCrafter Python package in:
    custom_nodes/ComfyUI-VizClick-NormalCrafter/vendor/normalcrafter/

Upstream:
https://github.com/Binyr/NormalCrafter
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .normalcrafter_node import VizClickNormalCrafter
    NODE_CLASS_MAPPINGS["VizClickNormalCrafter"] = VizClickNormalCrafter
    NODE_DISPLAY_NAME_MAPPINGS["VizClickNormalCrafter"] = "VizClick NormalCrafter"
except Exception as e:
    # Keep ComfyUI startup alive and expose the real error in the console.
    print(f"[VizClick NormalCrafter] Import failed: {type(e).__name__}: {e}")

try:
    from .normal_temporal_stabilizer import (
        VizClickNormalTemporalStabilizer,
        NODE_CLASS_MAPPINGS as _STAB_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _STAB_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_STAB_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_STAB_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    print(f"[VizClick NormalCrafter] Temporal stabilizer import failed: {type(e).__name__}: {e}")
