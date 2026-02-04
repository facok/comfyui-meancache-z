"""
File    : __init__.py
Purpose : MeanCache acceleration for Z-Image Flow Matching model
Author  : Based on UnicomAI MeanCache research
Date    : Jan 30, 2026
Repo    : https://unicomai.github.io/MeanCache/
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          comfyui-meancache-z
    MeanCache: From Instantaneous to Average Velocity for Accelerating
               Flow Matching Inference (UnicomAI)

    Training-free inference acceleration for Flow Matching models like
    Z-Image, FLUX.1, HunyuanVideo, etc.

    Core Algorithm:
    - Traditional: x_{t+dt} = x_t + dt * v(x_t, t)  [instantaneous velocity]
    - MeanCache:   x_{t+dt} = x_t + dt * v_avg      [JVP-corrected average velocity]

    Where: û(z_t, t, s) = v(z_t, t) + (s - t) · JVP_{r→t}  [paper formula]
           JVP_{r→t} ≈ (v_t - v_r) / (t - r)  [finite difference approximation]

    Features:
    - Training-free: No model fine-tuning required
    - JVP-based velocity correction for accurate ODE trajectory
    - PSSP (Peak-Suppressed Shortest Path) scheduling for optimal compute allocation
    - ~1.3-2.0x inference speedup while maintaining image quality
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""

# V3 ComfyExtension entry point
from comfy_api.latest import ComfyExtension, io
from .nodes.meancache_node import MeanCache_ZImage


class MeanCacheExtension(ComfyExtension):
    """
    MeanCache ComfyUI extension for Flow Matching model acceleration.
    """

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Return list of nodes provided by this extension.
        """
        return [
            MeanCache_ZImage,
        ]


async def comfy_entrypoint() -> MeanCacheExtension:
    """
    V3 async entry point for ComfyUI.
    """
    return MeanCacheExtension()


# V2 style exports for backwards compatibility
from .nodes.meancache_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serve frontend JavaScript (preset sync, widget control)
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "MeanCacheExtension",
    "comfy_entrypoint",
]
