"""
MeanCache_ZImage: Training-free inference acceleration for Z-Image Flow Matching models.

Based on UnicomAI MeanCache research:
"From Instantaneous to Average Velocity for Accelerating Flow Matching Inference"
https://unicomai.github.io/MeanCache/

This node patches Z-Image models to use average velocity (via JVP approximation)
instead of instantaneous velocity, enabling ~1.3-2.0x speedup while maintaining quality.
"""
from comfy_api.latest import io
from ..patch.model_patch import apply_meancache_to_model


# Preset acceleration profiles
# Each preset defines a tuned parameter set for different speed/quality tradeoffs.
# Tuned to match official MeanCache acceleration ratios.
PRESETS = {
    "Quality": {
        "rel_l1_thresh": 0.15,
        "skip_budget": 0.20,  # ~1.25x speedup, conservative
        "start_step": 3,
        "end_step": -1,
        "enable_pssp": True,
        "peak_threshold": 0.08,
        "gamma": 3.0,
        "adaptive_k": True,
    },
    "Balanced": {
        "rel_l1_thresh": 0.30,
        "skip_budget": 0.40,  # ~1.67x speedup, matches official 30-step
        "start_step": 2,
        "end_step": -1,
        "enable_pssp": True,
        "peak_threshold": 0.15,
        "gamma": 2.0,
        "adaptive_k": True,
    },
    "Speed": {
        "rel_l1_thresh": 0.50,
        "skip_budget": 0.55,  # ~2.2x speedup, matches official 22-step
        "start_step": 1,
        "end_step": -1,
        "enable_pssp": True,
        "peak_threshold": 0.35,
        "gamma": 1.5,
        "adaptive_k": True,
    },
    "Turbo": {
        "rel_l1_thresh": 0.55,
        "skip_budget": 0.60,  # ~2.5x speedup, aggressive but usable
        "start_step": 1,
        "end_step": -1,
        "enable_pssp": True,
        "peak_threshold": 0.45,
        "gamma": 1.0,
        "adaptive_k": True,
    },
}

PRESET_NAMES = ["Quality", "Balanced", "Speed", "Turbo", "Custom"]


class MeanCache_ZImage(io.ComfyNode):
    """
    Applies MeanCache acceleration to Z-Image Flow Matching models.

    MeanCache uses average velocity instead of instantaneous velocity
    to enable intelligent step skipping during ODE trajectory integration.

    Key Features:
    - Training-free: No model fine-tuning required
    - JVP-based velocity correction for accurate trajectory
    - PSSP scheduling for optimal compute budget allocation
    - Adaptive thresholding to prevent error accumulation
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MeanCache_ZImage",
            display_name="MeanCache (Z-Image)",
            category="model_patches/acceleration",
            description=(
                "Training-free inference acceleration for Z-Image using MeanCache. "
                "Computes average velocity via JVP approximation to enable intelligent "
                "step skipping while maintaining image quality. "
                "Based on UnicomAI research: https://unicomai.github.io/MeanCache/"
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The Z-Image model to accelerate with MeanCache."
                ),
                io.Combo.Input(
                    "preset",
                    options=PRESET_NAMES,
                    default="Balanced",
                    tooltip=(
                        "Acceleration preset. "
                        "Quality (~1.3x): conservative, minimal skipping. "
                        "Balanced (~1.5x): good speed/quality tradeoff. "
                        "Speed (~1.75x): aggressive skipping. "
                        "Turbo (~2.0x): maximum speed, may reduce quality. "
                        "Custom: use manual parameters below."
                    )
                ),
                io.Float.Input(
                    "rel_l1_thresh",
                    default=0.3,
                    min=0.05,
                    max=0.8,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "[Custom only] Relative L1 threshold for skip decision. "
                        "Lower = more quality, fewer skips. "
                        "Higher = more speedup, potential quality loss. "
                        "Recommended: 0.2-0.4"
                    )
                ),
                io.Float.Input(
                    "skip_budget",
                    default=0.3,
                    min=0.0,
                    max=0.75,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "[Custom only] Maximum fraction of steps to skip. "
                        "0.3 = up to 30% speedup potential. "
                        "Set to 0.0 to disable skipping."
                    )
                ),
                io.Int.Input(
                    "start_step",
                    default=2,
                    min=0,
                    max=20,
                    step=1,
                    optional=True,
                    tooltip=(
                        "[Custom only] Step index to begin caching (0-indexed). "
                        "Early steps form image structure and should not be skipped."
                    )
                ),
                io.Int.Input(
                    "end_step",
                    default=-1,
                    min=-1,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "[Custom only] Step index to stop caching. "
                        "-1 = cache until the end."
                    )
                ),
                io.Boolean.Input(
                    "enable_pssp",
                    default=True,
                    optional=True,
                    label_on="PSSP Scheduling Enabled",
                    label_off="Simple Threshold Mode",
                    tooltip=(
                        "[Custom only] Enable Peak-Suppressed Shortest Path scheduling. "
                        "PSSP optimally allocates compute budget across steps."
                    )
                ),
                io.Float.Input(
                    "peak_threshold",
                    default=0.15,
                    min=0.05,
                    max=0.60,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "[Custom only] Maximum allowed single-step velocity deviation. "
                        "Lower = more conservative, higher = more aggressive."
                    )
                ),
                io.Boolean.Input(
                    "adaptive_k",
                    default=True,
                    optional=True,
                    label_on="Adaptive K Enabled",
                    label_off="Fixed K Mode",
                    tooltip=(
                        "[Custom only] Dynamically select JVP lookback steps K based on sigma. "
                        "Early steps use small K (captures rapid changes), "
                        "later steps use large K (smoother estimates). "
                        "Based on official MeanCache edge_order patterns."
                    )
                ),
                io.Combo.Input(
                    "cache_device",
                    options=["cpu", "cuda"],
                    default="cpu",
                    tooltip=(
                        "Device for storing velocity cache. "
                        "'cpu' = saves VRAM, slight transfer overhead. "
                        "'cuda' = faster access but uses more VRAM."
                    )
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    optional=True,
                    label_on="Debug Logging On",
                    label_off="Debug Logging Off",
                    tooltip=(
                        "Enable debug logging to console. "
                        "Shows per-step skip/compute decisions and L_K values."
                    )
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="Model with MeanCache acceleration applied. Use with any compatible sampler."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        preset: str = "Balanced",
        rel_l1_thresh: float = 0.3,
        skip_budget: float = 0.3,
        start_step: int = 2,
        end_step: int = -1,
        enable_pssp: bool = True,
        peak_threshold: float = 0.15,
        adaptive_k: bool = True,
        cache_device: str = "cpu",
        debug: bool = False,
    ) -> io.NodeOutput:
        """
        Apply MeanCache to the model and return patched version.
        """
        # Resolve parameters from preset or custom values
        if preset in PRESETS:
            params = PRESETS[preset]
            rel_l1_thresh = params["rel_l1_thresh"]
            skip_budget = params["skip_budget"]
            start_step = params["start_step"]
            end_step = params["end_step"]
            enable_pssp = params["enable_pssp"]
            peak_threshold = params["peak_threshold"]
            gamma = params["gamma"]
            adaptive_k = params["adaptive_k"]
        else:
            # Custom mode: use manual parameter values
            gamma = 2.0

        patched_model = apply_meancache_to_model(
            model=model,
            rel_l1_thresh=rel_l1_thresh,
            skip_budget=skip_budget,
            start_step=start_step,
            end_step=end_step,
            cache_device=cache_device,
            enable_pssp=enable_pssp,
            peak_threshold=peak_threshold,
            gamma=gamma,
            adaptive_k=adaptive_k,
            debug=debug,
            preset_name=preset,
        )

        return io.NodeOutput(patched_model)


# V2 style node registration for backwards compatibility
NODE_CLASS_MAPPINGS = {
    "MeanCache_ZImage": MeanCache_ZImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeanCache_ZImage": "MeanCache (Z-Image)",
}
