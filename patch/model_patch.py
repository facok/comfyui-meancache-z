"""
MeanCache model patching for Z-Image Flow Matching models.

This module applies MeanCache acceleration by wrapping the model's forward pass
using ComfyUI's set_model_unet_function_wrapper mechanism.

Reference implementations:
- DyPE: ComfyUI-DyPE/src/patch_utils.py
- TeaCache: ComfyUI-WanVideoWrapper/cache_methods/cache_methods.py
"""
import torch
import hashlib
from typing import Dict, Any, Callable, Optional
from comfy.model_patcher import ModelPatcher
from comfy import model_management as mm

from ..core.meancache_state import MeanCacheState, DEFAULT_MAX_CACHE_SPAN
from ..core.velocity_cache import (
    compute_jvp_approximation,
    compute_velocity_similarity,
    compute_online_L_K,
    should_skip_step,
    get_optimal_k
)
from ..core.trajectory_scheduler import TrajectoryScheduler

# Module-level debug flag, controlled by node's debug parameter
MEANCACHE_DEBUG = False

def _debug_log(msg: str):
    """Print debug message if debug mode is enabled."""
    if MEANCACHE_DEBUG:
        print(f"[MeanCache] {msg}")


def _get_lora_fingerprint(model_patcher: ModelPatcher) -> str:
    """
    Generate a fingerprint for model patch state (including LoRA) to detect changes.

    When LoRA is applied/removed/changed, the model's patches dict changes,
    making cached velocities invalid. This function creates a fingerprint
    based on the actual applied patches, which is more reliable than
    checking model_options (which may not contain LoRA metadata).

    Args:
        model_patcher: The ComfyUI ModelPatcher instance

    Returns:
        String fingerprint of patch state ("no_patches" if none applied)
    """
    try:
        # Get the actual patches dict from ModelPatcher
        # This contains all applied weight patches including LoRA
        patches = getattr(model_patcher, 'patches', None)
        if not patches:
            return "no_patches"

        # Create fingerprint from patch keys and strengths
        # Format: hash of "key1:strength1|key2:strength2|..."
        patch_info = []
        for key in sorted(patches.keys()):
            patch_data = patches[key]
            # patch_data is typically (strength, (lora_key, weight, ...))
            if isinstance(patch_data, tuple) and len(patch_data) >= 1:
                strength = patch_data[0]
                patch_info.append(f"{key}:{strength:.6f}")
            else:
                patch_info.append(str(key))

        if not patch_info:
            return "no_patches"

        # Use hash for compact fingerprint
        fingerprint_str = "|".join(patch_info)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    except Exception:
        # Fallback: if we can't read patches, return a safe default
        # that won't cause unnecessary resets
        return "unknown"


def apply_meancache_to_model(
    model: ModelPatcher,
    rel_l1_thresh: float = 0.3,
    skip_budget: float = 0.3,
    start_step: int = 2,
    end_step: int = -1,
    cache_device: str = 'cpu',
    enable_pssp: bool = True,
    peak_threshold: float = 0.15,
    gamma: float = 2.0,
    max_cache_span: int = DEFAULT_MAX_CACHE_SPAN,
    adaptive_k: bool = True,
    debug: bool = False,
    preset_name: str = "Custom",
) -> ModelPatcher:
    """
    Apply MeanCache acceleration to a Z-Image model.

    This patches the model to use average velocity (via JVP correction)
    instead of instantaneous velocity during sampling, enabling intelligent
    step skipping while maintaining image quality.

    Paper: "From Instantaneous to Average Velocity for Accelerating
           Flow Matching Inference" (UnicomAI MeanCache)

    Args:
        model: Input ModelPatcher to patch
        rel_l1_thresh: Relative L1 threshold for skip decision (lower=quality, higher=speed)
        skip_budget: Maximum fraction of steps to skip (0.0-0.5)
        start_step: Step index to start caching (skip early structure steps)
        end_step: Step index to end caching (-1 = cache until end)
        cache_device: Device for velocity cache ('cpu' saves VRAM, 'cuda' faster)
        enable_pssp: Enable Peak-Suppressed Shortest Path scheduling
        peak_threshold: Maximum allowed single-step deviation for PSSP
        gamma: PSSP penalty exponent (higher = penalize large deviations more)
        max_cache_span: Maximum cache span K for JVP_K computation
        adaptive_k: Dynamically select K based on sigma (True) or use max K (False)
        debug: Enable debug logging to console
        preset_name: Name of the active preset (for summary display)

    Returns:
        Patched ModelPatcher with MeanCache wrapper applied
    """
    # Set module-level debug flag
    global MEANCACHE_DEBUG
    MEANCACHE_DEBUG = debug

    # Clone model to avoid modifying original
    m = model.clone()

    # Resolve cache device
    if cache_device == 'cuda':
        cache_device_resolved = mm.get_torch_device()
    else:
        cache_device_resolved = 'cpu'

    # Initialize state manager with cache span support
    meancache_state = MeanCacheState(
        cache_device=cache_device_resolved,
        max_cache_span=max_cache_span
    )

    # Store configuration in model options for access during sampling
    m.model_options.setdefault("meancache", {})
    m.model_options["meancache"]["state"] = meancache_state
    m.model_options["meancache"]["config"] = {
        "rel_l1_thresh": rel_l1_thresh,
        "skip_budget": skip_budget,
        "start_step": start_step,
        "end_step": end_step,
        "enable_pssp": enable_pssp,
        "peak_threshold": peak_threshold,
        "cache_device": cache_device_resolved,
        "gamma": gamma,
        "max_cache_span": max_cache_span,
        "adaptive_k": adaptive_k,
        "preset_name": preset_name,
    }

    # Get sigma schedule info for normalization
    try:
        sigma_max = m.model.model_sampling.sigma_max.item()
    except:
        sigma_max = 1.0

    # Create the wrapper function that will intercept model calls
    def meancache_wrapper_function(
        model_function: Callable,
        args_dict: Dict[str, Any]
    ) -> torch.Tensor:
        """
        MeanCache wrapper that intercepts every model forward pass.

        This wrapper:
        1. Checks if current step is in active range
        2. Decides whether to skip based on velocity similarity
        3. Computes JVP-corrected average velocity for better trajectory
        4. Caches velocities for potential skip reuse
        """
        input_x = args_dict.get("input")
        timestep = args_dict.get("timestep")
        c = args_dict.get("c", {})

        # Get configuration
        config = m.model_options["meancache"]["config"]
        state = m.model_options["meancache"]["state"]

        # Get transformer options for step tracking
        transformer_options = c.get("transformer_options", {})

        # Get current sigma value
        if timestep is not None and timestep.numel() > 0:
            current_sigma = timestep.flatten()[0].item()
        else:
            current_sigma = 1.0

        # Determine effective end step
        effective_end = config["end_step"] if config["end_step"] >= 0 else 999999

        # Compute total steps from sigma schedule
        sigmas = transformer_options.get("sample_sigmas")
        total_steps = len(sigmas) - 1 if sigmas is not None else 20

        # Get or create scheduler for PSSP
        # Store scheduler in meancache_state to persist across steps
        scheduler = meancache_state.scheduler
        if scheduler is None and config["enable_pssp"]:
            scheduler = TrajectoryScheduler(
                total_steps=total_steps,
                skip_budget=config["skip_budget"],
                peak_threshold=config["peak_threshold"],
                gamma=config["gamma"]
            )
            # Adjust schedule based on actual sigma values if available
            if sigmas is not None:
                scheduler.adjust_for_sigmas(sigmas)
            meancache_state.scheduler = scheduler

        # Process the batch - handle CFG (conditional/unconditional)
        batch_size = input_x.shape[0]

        # Detect patch state changes (including LoRA) - if patches change,
        # cached velocities are invalid for different model weights
        current_patch_key = _get_lora_fingerprint(m)
        last_patch_key = getattr(meancache_state, '_last_patch_key', None)
        if last_patch_key is not None and current_patch_key != last_patch_key:
            _debug_log(f"Model patches changed, resetting MeanCache state")
            state.clear_all()
            meancache_state.scheduler = None
            meancache_state._summary_printed = False
        meancache_state._last_patch_key = current_patch_key

        # Detect new sampling run by sigma jump (handles ComfyUI model caching)
        # During sampling, sigma decreases monotonically. A jump UP means new run started.
        last_sigma = meancache_state.last_sigma
        if last_sigma is not None and current_sigma > last_sigma + 0.1:
            _debug_log(f"New sampling run detected (sigma jump {last_sigma:.4f} → {current_sigma:.4f}), resetting state")
            state.clear_all()
            meancache_state.last_sigma = None
            meancache_state._last_patch_key = None  # Force re-detection of patches for new run
            meancache_state._summary_printed = False
            last_sigma = None  # Update local variable too

        # Detect cond vs uncond for split CFG (batch_size=1, called twice per step)
        # If sigma is same as last call, this is the second call (uncond) → pred_id=1
        if batch_size == 1 and last_sigma is not None and abs(current_sigma - last_sigma) < 1e-6:
            split_pred_id = 1  # uncond call
        else:
            split_pred_id = 0  # cond call (or no CFG)
        meancache_state.last_sigma = current_sigma

        # Get step index from the target prediction's own state (before increment)
        # This avoids off-by-one when pred_id=0 already incremented before uncond call
        if batch_size == 1:
            pred_state_target = state.get_or_create(split_pred_id, config["cache_device"])
            step_index = pred_state_target.get('step_index', 0)
        else:
            pred_state_ref = state.get_or_create(0, config["cache_device"])
            step_index = pred_state_ref.get('step_index', 0)

        # Check if we're in active caching range
        in_active_range = config["start_step"] <= step_index < effective_end

        # For split CFG: compute unified error so cond/uncond make the SAME skip decision
        # Without this, marginal differences (e.g. 0.0434 vs 0.0466) cause one to skip
        # and the other to compute, creating CFG artifacts
        unified_error = None
        if batch_size == 1:
            if split_pred_id == 0:
                # First call of this step: snapshot worst-case error from both pred_ids
                err_0 = state.get_or_create(0, config["cache_device"]).get('accumulated_error', 1.0)
                err_1 = state.get_or_create(1, config["cache_device"]).get('accumulated_error', 1.0)
                meancache_state.split_cfg_unified_error = max(err_0, err_1)
            # Both pred_ids use the same unified error (set once per step by pred_id=0)
            unified_error = meancache_state.split_cfg_unified_error

        _debug_log(f"Step {step_index}: sigma={current_sigma:.4f}, in_active_range={in_active_range}, batch={batch_size}" +
                   (f", split_pred_id={split_pred_id}" if batch_size == 1 else ""))

        if batch_size == 1:
            # Single prediction - use detected pred_id for split CFG
            output = _process_single_prediction(
                model_function=model_function,
                input_x=input_x,
                timestep=timestep,
                c=c,
                pred_id=split_pred_id,
                state=state,
                config=config,
                scheduler=scheduler,
                step_index=step_index,
                current_sigma=current_sigma,
                in_active_range=in_active_range,
                unified_accumulated_error=unified_error
            )
        else:
            # Batched CFG - process full batch but track state per-type
            output = _process_batch_prediction(
                model_function=model_function,
                input_x=input_x,
                timestep=timestep,
                c=c,
                batch_size=batch_size,
                state=state,
                config=config,
                scheduler=scheduler,
                step_index=step_index,
                current_sigma=current_sigma,
                in_active_range=in_active_range,
            )

        # Print acceleration summary when sampling completes
        if not getattr(meancache_state, '_summary_printed', False):
            all_done = len(state.states) > 0
            for pid in state.states:
                if state.states[pid].get('step_index', 0) < total_steps:
                    all_done = False
                    break
            if all_done:
                meancache_state._summary_printed = True
                # Use pred_id=0 (cond) as representative
                pred0 = state.get(0)
                skip_count = state.get_skip_count(0)
                total = pred0.get('step_index', 0)
                compute_count = total - skip_count
                skip_pct = (skip_count / total * 100) if total > 0 else 0
                speedup = total / compute_count if compute_count > 0 else 1.0
                preset = config.get("preset_name", "Custom")
                print(f"[MeanCache] Sampling complete ({preset}): {total} steps, "
                      f"{skip_count} skipped, {compute_count} computed "
                      f"({skip_pct:.1f}% skip rate, ~{speedup:.2f}x speedup)")

        return output

    # Set the wrapper using ComfyUI's model patcher API
    m.set_model_unet_function_wrapper(meancache_wrapper_function)

    _debug_log(f"MeanCache applied: thresh={rel_l1_thresh}, budget={skip_budget}, start={start_step}, end={end_step}, pssp={enable_pssp}")

    return m


def _process_single_prediction(
    model_function: Callable,
    input_x: torch.Tensor,
    timestep: torch.Tensor,
    c: Dict[str, Any],
    pred_id: int,
    state: MeanCacheState,
    config: Dict[str, Any],
    scheduler: Optional[TrajectoryScheduler],
    step_index: int,
    current_sigma: float,
    in_active_range: bool,
    unified_accumulated_error: Optional[float] = None
) -> torch.Tensor:
    """
    Process a single prediction with MeanCache logic.

    Implements the paper's core algorithm:
    1. Compute JVP_K using multi-step velocity history
    2. Estimate average velocity: û = v + dt * JVP
    3. Use stability deviation L_K for skip decision
    4. Cache velocities for trajectory optimization

    Args:
        model_function: Original model forward function
        input_x: Input latent tensor
        timestep: Current timestep/sigma
        c: Conditioning dictionary
        pred_id: Prediction ID (0=cond, 1=uncond)
        state: MeanCache state manager
        config: MeanCache configuration
        scheduler: Optional PSSP scheduler
        step_index: Current step index
        current_sigma: Current sigma value
        in_active_range: Whether we're in active caching range

    Returns:
        Model output (possibly from cache)
    """
    # Get or create state for this prediction
    pred_state = state.get_or_create(pred_id, config["cache_device"])

    # Determine if we should skip
    should_skip = False
    # Use unified error for split CFG (ensures cond/uncond make same decision)
    # Fall back to per-pred error for batch mode or non-CFG
    if unified_accumulated_error is not None:
        accumulated_error = unified_accumulated_error
    else:
        accumulated_error = pred_state.get('accumulated_error', 1.0)

    if in_active_range and pred_state.get('v_cache') is not None:
        # Use accumulated error (from stability deviation) for skip decision
        if scheduler is not None:
            # Use PSSP scheduler with stability deviation metric
            should_skip, accumulated_error = scheduler.get_skip_decision(
                step_index=step_index,
                velocity_similarity=accumulated_error,
                accumulated_error=accumulated_error
            )
        else:
            # Simple threshold check using accumulated error
            should_skip, accumulated_error = should_skip_step(
                stability_deviation=accumulated_error,
                threshold=config["rel_l1_thresh"],
                accumulated_error=accumulated_error
            )
        state.update(pred_id, accumulated_error=accumulated_error)

    # Execute based on decision
    if should_skip and pred_state.get('v_cache') is not None:
        # SKIP: Apply JVP correction to estimate current velocity (paper's core algorithm)
        # Formula: û(z_t, t, s) = v(z_t, t) + (s - t) · JVP
        v_cache = pred_state['v_cache'].to(input_x.device)
        jvp_cache = pred_state.get('jvp_cache')
        sigma_cache = pred_state.get('sigma_cache', current_sigma)

        if jvp_cache is not None:
            # Apply JVP correction: estimate current velocity from cached data
            # NOTE: dt is positive (sigma_cache > current_sigma), but JVP formula requires
            # addition because the correction direction matches velocity evolution
            dt = sigma_cache - current_sigma
            jvp_device = jvp_cache.to(input_x.device)
            output = v_cache + dt * jvp_device
            _debug_log(f"  pred_id={pred_id}: SKIP+JVP (dt={dt:.4f}, deviation={accumulated_error:.4f})")
        else:
            # Fallback: use cached velocity directly (less accurate)
            output = v_cache
            _debug_log(f"  pred_id={pred_id}: SKIP (no JVP, deviation={accumulated_error:.4f})")

        state.record_skip(pred_id, step_index)

        # Update history with estimated velocity (matches official implementation behavior)
        # This ensures v_history grows even during skips for accurate JVP_K computation
        state.update_history(pred_id, output.detach(), current_sigma)
    else:
        # COMPUTE: Full model forward pass
        output = model_function(input_x, timestep, **c)
        _debug_log(f"  pred_id={pred_id}: COMPUTE (in_range={in_active_range}, has_cache={pred_state.get('v_cache') is not None})")

        v_current = output

        # Update velocity history for JVP_K computation
        state.update_history(pred_id, v_current, current_sigma)

        # Try to compute JVP using multi-step history (JVP_K)
        # Adaptive K: select optimal K based on sigma (official MeanCache pattern)
        # Fixed K: use largest available K (original behavior)
        jvp = None
        if config.get("adaptive_k", True):
            # Adaptive: early steps use small K (fast changes), late steps use large K (stable)
            optimal_k = get_optimal_k(current_sigma, config["max_cache_span"])
            jvp = state.get_jvp_k(pred_id, optimal_k)
            # Fallback to smaller K if optimal not available
            if jvp is None:
                for k in range(min(optimal_k - 1, config["max_cache_span"]), 0, -1):
                    jvp_k = state.get_jvp_k(pred_id, k)
                    if jvp_k is not None:
                        jvp = jvp_k
                        break
        else:
            # Fixed: try largest K first (original behavior)
            for k in range(config["max_cache_span"], 0, -1):
                jvp_k = state.get_jvp_k(pred_id, k)
                if jvp_k is not None:
                    jvp = jvp_k
                    break

        # Fall back to simple two-step JVP if history-based JVP not available
        if jvp is None:
            v_prev = pred_state.get('v_prev')
            t_prev = pred_state.get('t_prev')
            if v_prev is not None and t_prev is not None:
                dt_prev = t_prev - current_sigma  # sigma decreases over time
                if abs(dt_prev) > 1e-8:
                    jvp = compute_jvp_approximation(v_current, v_prev, dt_prev)

        # Compute deviation for adaptive skip decision using paper's L_K metric
        # L_K measures retrospective JVP accuracy: how well did JVP extrapolation predict v_current?
        v_cache_old = pred_state.get('v_cache')
        jvp_cache_old = pred_state.get('jvp_cache')
        sigma_cache_old = pred_state.get('sigma_cache')

        if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
            # Paper's L_K: retrospective JVP accuracy
            dt_elapsed = sigma_cache_old - current_sigma
            deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
            _debug_log(f"  pred_id={pred_id}: L_K={deviation:.4f} (threshold={config['rel_l1_thresh']})")
        elif v_cache_old is not None:
            # Fallback: relative L1 (first few steps before JVP available)
            deviation = compute_velocity_similarity(v_current, v_cache_old)
            _debug_log(f"  pred_id={pred_id}: rel_L1={deviation:.4f} (no JVP yet)")
        else:
            deviation = 1.0  # Force compute on first step

        state.update(pred_id, accumulated_error=deviation)

        # Cache current velocity, JVP, and sigma for potential future skip with JVP correction
        state.update(pred_id, v_cache=v_current.to(config["cache_device"]))
        state.update(pred_id, sigma_cache=current_sigma)
        if jvp is not None:
            state.update(pred_id, jvp_cache=jvp.to(config["cache_device"]))

        # Update state for next iteration (legacy support)
        state.update(
            pred_id,
            v_prev=v_current.to(config["cache_device"]),
            t_prev=current_sigma,
        )

    # Increment step counter
    state.increment_step(pred_id)

    return output


def _process_batch_prediction(
    model_function: Callable,
    input_x: torch.Tensor,
    timestep: torch.Tensor,
    c: Dict[str, Any],
    batch_size: int,
    state: MeanCacheState,
    config: Dict[str, Any],
    scheduler: Optional[TrajectoryScheduler],
    step_index: int,
    current_sigma: float,
    in_active_range: bool,
) -> torch.Tensor:
    """
    Process batched CFG prediction.

    For CFG, the model receives [cond, uncond] batched together.
    We process the full batch but track state per-prediction type.

    Args:
        model_function: Original model forward function
        input_x: Batched input latent tensor
        timestep: Batched timestep tensor
        c: Conditioning dictionary
        batch_size: Batch size (typically 2 for CFG)
        state: MeanCache state manager
        config: MeanCache configuration
        scheduler: Optional PSSP scheduler
        step_index: Current step index
        current_sigma: Current sigma value
        in_active_range: Whether we're in active caching range

    Returns:
        Batched model output
    """
    # Check if ALL predictions can be skipped
    can_skip_all = True
    outputs = []

    for pred_id in range(batch_size):
        pred_state = state.get_or_create(pred_id, config["cache_device"])

        if not in_active_range or pred_state.get('v_cache') is None:
            can_skip_all = False
            break

        accumulated_error = pred_state.get('accumulated_error', 1.0)

        if scheduler is not None:
            should_skip, _ = scheduler.get_skip_decision(
                step_index=step_index,
                velocity_similarity=accumulated_error,
                accumulated_error=accumulated_error
            )
        else:
            should_skip, _ = should_skip_step(
                stability_deviation=accumulated_error,
                threshold=config["rel_l1_thresh"],
                accumulated_error=accumulated_error
            )

        if not should_skip:
            can_skip_all = False
            break

    if can_skip_all:
        # All predictions can be skipped - apply JVP correction (paper's core algorithm)
        _debug_log(f"  BATCH SKIP+JVP (batch_size={batch_size})")
        for pred_id in range(batch_size):
            pred_state = state.get(pred_id)
            v_cache = pred_state['v_cache'].to(input_x.device)
            jvp_cache = pred_state.get('jvp_cache')
            sigma_cache = pred_state.get('sigma_cache', current_sigma)

            if jvp_cache is not None:
                # Apply JVP correction
                # NOTE: dt is positive, but JVP formula requires addition
                dt = sigma_cache - current_sigma
                jvp_device = jvp_cache.to(input_x.device)
                v_corrected = v_cache + dt * jvp_device
                outputs.append(v_corrected)
            else:
                outputs.append(v_cache)

            state.record_skip(pred_id, step_index)
            state.increment_step(pred_id)

        # Update history with estimated velocities for all skipped predictions
        for pred_id in range(batch_size):
            v_estimated = outputs[pred_id]
            state.update_history(pred_id, v_estimated.detach(), current_sigma)

        return torch.cat(outputs, dim=0)

    # At least one prediction needs computation - run full batch
    _debug_log(f"  BATCH COMPUTE (batch_size={batch_size})")
    output = model_function(input_x, timestep, **c)

    # Update state for each prediction in batch
    for pred_id in range(batch_size):
        pred_state = state.get_or_create(pred_id, config["cache_device"])

        v_current = output[pred_id:pred_id+1]

        # Update velocity history for JVP_K computation
        state.update_history(pred_id, v_current, current_sigma)

        # Try to compute JVP using multi-step history
        # Adaptive K: select optimal K based on sigma (official MeanCache pattern)
        # Fixed K: use largest available K (original behavior)
        jvp = None
        if config.get("adaptive_k", True):
            optimal_k = get_optimal_k(current_sigma, config["max_cache_span"])
            jvp = state.get_jvp_k(pred_id, optimal_k)
            if jvp is None:
                for k in range(min(optimal_k - 1, config["max_cache_span"]), 0, -1):
                    jvp_k = state.get_jvp_k(pred_id, k)
                    if jvp_k is not None:
                        jvp = jvp_k
                        break
        else:
            for k in range(config["max_cache_span"], 0, -1):
                jvp_k = state.get_jvp_k(pred_id, k)
                if jvp_k is not None:
                    jvp = jvp_k
                    break

        # Fall back to simple two-step JVP
        if jvp is None:
            v_prev = pred_state.get('v_prev')
            t_prev = pred_state.get('t_prev')
            if v_prev is not None and t_prev is not None:
                dt_prev = t_prev - current_sigma
                if abs(dt_prev) > 1e-8:
                    jvp = compute_jvp_approximation(v_current, v_prev, dt_prev)

        # Compute deviation for adaptive skip decision using paper's L_K metric
        v_cache_old = pred_state.get('v_cache')
        jvp_cache_old = pred_state.get('jvp_cache')
        sigma_cache_old = pred_state.get('sigma_cache')

        if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
            # Paper's L_K: retrospective JVP accuracy
            dt_elapsed = sigma_cache_old - current_sigma
            deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
            _debug_log(f"  batch pred_id={pred_id}: L_K={deviation:.4f}")
        elif v_cache_old is not None:
            # Fallback: relative L1 (first few steps before JVP available)
            deviation = compute_velocity_similarity(v_current, v_cache_old)
            _debug_log(f"  batch pred_id={pred_id}: rel_L1={deviation:.4f}")
        else:
            deviation = 1.0

        state.update(pred_id, accumulated_error=deviation)

        # Cache current velocity, JVP, and sigma for potential future skip with JVP correction
        state.update(pred_id, v_cache=v_current.to(config["cache_device"]))
        state.update(pred_id, sigma_cache=current_sigma)
        if jvp is not None:
            state.update(pred_id, jvp_cache=jvp.to(config["cache_device"]))

        # Update state for next iteration
        state.update(
            pred_id,
            v_prev=v_current.to(config["cache_device"]),
            t_prev=current_sigma,
        )
        state.increment_step(pred_id)

    return output


def get_meancache_report(model: ModelPatcher) -> Optional[Dict[str, Any]]:
    """
    Get MeanCache performance report from a patched model.

    Args:
        model: Model that was patched with apply_meancache_to_model

    Returns:
        Report dictionary or None if not a MeanCache model
    """
    meancache_opts = model.model_options.get("meancache")
    if meancache_opts is None:
        return None

    state = meancache_opts.get("state")
    if state is None:
        return None

    return state.get_report()


def clear_meancache_state(model: ModelPatcher) -> None:
    """
    Clear MeanCache state from a patched model.

    Call this between generations to reset state.

    Args:
        model: Model that was patched with apply_meancache_to_model
    """
    meancache_opts = model.model_options.get("meancache")
    if meancache_opts is not None:
        state = meancache_opts.get("state")
        if state is not None:
            state.clear_all()


