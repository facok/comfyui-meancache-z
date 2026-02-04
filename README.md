# MeanCache for Z-Image (ComfyUI)

Training-free inference acceleration for Z-Image Flow Matching models based on [UnicomAI MeanCache](https://unicomai.github.io/MeanCache/).
<img width="2112" height="1148" alt="z_00128_" src="https://github.com/user-attachments/assets/89fc1acf-390c-4c87-ae42-a7071712bf31" />

## Features

- **Training-free**: No model fine-tuning required
- **JVP-based velocity correction**: Uses average velocity instead of instantaneous velocity for accurate ODE trajectory
- **PSSP scheduling**: Peak-Suppressed Shortest Path algorithm for optimal compute budget allocation
- **Preset profiles**: Quality / Balanced / Speed / Turbo presets for easy configuration
- **~1.4x-2.0x speedup**: Inference acceleration while maintaining image quality

## Installation

Copy the `comfyui-meancache-z` folder to your ComfyUI `custom_nodes` directory.

## Usage

1. Load your Z-Image model
2. Connect it to the **MeanCache (Z-Image)** node
3. Select a preset or use Custom mode
4. Connect the patched model output to your sampler
<img width="1182" height="840" alt="image" src="https://github.com/user-attachments/assets/539c4bfd-5499-43b2-81a1-be58a9eb55fd" />

### Presets

| Preset | Speedup | Description |
|--------|---------|-------------|
| Quality | ~1.25x | Conservative, minimal skipping, best quality |
| Balanced | ~1.7x | Good speed/quality tradeoff (default) |
| Speed | ~1.75x | Aggressive skipping |
| Turbo | ~2.0x | Maximum speed, may reduce quality |
| Custom | - | Manual parameter control |

### Preset Parameters

| Preset | rel_l1_thresh | skip_budget | start_step | peak_threshold | gamma |
|--------|---------------|-------------|------------|----------------|-------|
| Quality | 0.15 | 0.20 | 3 | 0.08 | 3.0 |
| Balanced | 0.30 | 0.40 | 2 | 0.15 | 2.0 |
| Speed | 0.50 | 0.55 | 1 | 0.35 | 1.5 |
| Turbo | 0.55 | 0.60 | 1 | 0.45 | 1.0 |

### Custom Mode Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| rel_l1_thresh | 0.3 | 0.05-0.70 | Skip threshold (lower=quality, higher=speed) |
| skip_budget | 0.3 | 0.0-0.75 | Max fraction of steps to skip |
| start_step | 2 | 0-20 | Step to begin caching (protect early structure) |
| end_step | -1 | -1 or 0+ | Step to end caching (-1=until end) |
| enable_pssp | True | - | Enable PSSP trajectory scheduling |
| peak_threshold | 0.15 | 0.05-0.60 | Max single-step velocity deviation |
| gamma | 2.0 | 0.5-3.0 | PSSP peak suppression exponent |
| cache_device | cpu | cpu/cuda | Device for velocity cache |
| debug | False | - | Enable debug logging |

## Sampling Summary

When sampling completes, a summary is printed to console:

```
[MeanCache] Sampling complete (Balanced): 35 steps, 14 skipped, 21 computed (40.0% skip rate, ~1.67x speedup)
```

## Algorithm

MeanCache improves Flow Matching inference by:

1. Computing JVP (Jacobian-Vector Product) approximation via finite differences:
   ```
   JVP_{r→t} ≈ (v_t - v_r) / (t - r)
   ```

2. Using average velocity instead of instantaneous velocity:
   ```
   û(z_t, t, s) = v(z_t, t) + (s - t) · JVP_{r→t}
   ```

3. Stability deviation metric (L_K) for adaptive skip decision:
   ```
   L_K = ||v_current - (v_prev + dt · JVP)|| / ||v_current||
   ```

4. PSSP scheduling with dynamic programming:
   ```
   π* = argmin Σ C(e)^γ   s.t. |π| ≤ B
   ```

5. Intelligently skipping steps when velocity is stable, using cached JVP-corrected velocity

## File Structure

```
comfyui-meancache-z/
├── __init__.py              # Plugin entry point (V2/V3 compatible)
├── nodes/
│   └── meancache_node.py    # MeanCache_ZImage node definition
├── patch/
│   └── model_patch.py       # Model wrapper with MeanCache logic
├── core/
│   ├── meancache_state.py   # State management per prediction
│   ├── velocity_cache.py    # JVP computation utilities
│   └── trajectory_scheduler.py  # PSSP scheduling algorithm
├── web/js/
│   └── meancache_preset.js  # Frontend preset widget sync
└── README.md
```

## References

- [MeanCache Paper](https://unicomai.github.io/MeanCache/) (UnicomAI)
- "From Instantaneous to Average Velocity for Accelerating Flow Matching Inference"

## License

MIT License
