"""
MeanCache velocity caching and JVP (Jacobian-Vector Product) computation.

Core algorithm from UnicomAI MeanCache paper:
"From Instantaneous to Average Velocity for Accelerating Flow Matching Inference"

Key formulas:
- JVP approximation: JVP_{r→t} ≈ (v_t - v_r) / (t - r)
- Average velocity: û(z_t, t, s) = v(z_t, t) + (s - t) · JVP_{r→t}
- Stability deviation: L_K(t,s) = (1/N) ||u(z_t,t,s) - v(z_t,t) - (s-t)·JVP_K||_1

Reference: https://unicomai.github.io/MeanCache/
"""
import torch
from typing import Tuple, Optional


def get_optimal_k(sigma: float, max_k: int = 5) -> int:
    """
    根据 sigma 值动态选择最优 JVP 回溯步数 K。

    基于官方 MeanCache edge_order 模式分析:
    - 早中期 (sigma > 0.5): 速度变化快，短回溯更准确
    - 中期 (sigma 0.2-0.5): 适度平滑
    - 后期 (sigma < 0.2): 速度稳定，长回溯更平滑

    Args:
        sigma: 当前 sigma 值 (噪声水平)
        max_k: 最大允许的 K 值

    Returns:
        最优 K 值
    """
    if sigma > 0.5:
        return 1  # 早中期: 速度变化快，短回溯
    elif sigma > 0.2:
        return min(max_k, 2)  # 中期: 适度平滑
    elif sigma > 0.1:
        return min(max_k, 3)  # 中后期: 较长回溯
    else:
        return max_k  # 后期: 最大平滑


def compute_jvp_approximation(
    v_current: torch.Tensor,
    v_prev: torch.Tensor,
    dt_prev: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Approximate Jacobian-vector product using finite differences.

    JVP_{r→t} describes how the velocity field changes over time interval [r, t].
    This allows estimating trajectory evolution without additional model evaluations.

    Formula: JVP_{r→t} ≈ (v_t - v_r) / (t - r)

    Args:
        v_current: Current velocity prediction v_t
        v_prev: Previous velocity prediction v_r (from timestep r)
        dt_prev: Time delta from previous step (t - r), typically sigma_r - sigma_t
        eps: Small value to prevent division by zero

    Returns:
        Approximated JVP tensor with same shape as v_current
    """
    if abs(dt_prev) < eps:
        return torch.zeros_like(v_current)

    v_prev_device = v_prev.to(v_current.device)
    return (v_current - v_prev_device) / dt_prev


def compute_jvp_k(
    v_history: list,
    t_history: list,
    k: int,
    eps: float = 1e-8
) -> Optional[torch.Tensor]:
    """
    Compute JVP with cache span K (using velocity from K steps ago).

    JVP_K estimates velocity change rate over K timesteps, which can be
    more stable than single-step JVP for larger skip intervals.

    Formula: JVP_K = (v_now - v_{now-K}) / (t_now - t_{now-K})

    Args:
        v_history: List of recent velocity tensors [v_{t-K}, ..., v_t]
        t_history: List of corresponding timesteps [t_{t-K}, ..., t_t]
        k: Cache span (number of steps to look back)
        eps: Small value to prevent division by zero

    Returns:
        JVP_K tensor or None if insufficient history
    """
    if len(v_history) < k + 1 or len(t_history) < k + 1:
        return None

    v_now = v_history[-1]
    v_k_ago = v_history[-(k + 1)]
    t_now = t_history[-1]
    t_k_ago = t_history[-(k + 1)]

    dt = t_k_ago - t_now  # sigma decreases, so t_k_ago > t_now
    if abs(dt) < eps:
        return None

    return (v_now - v_k_ago.to(v_now.device)) / dt


def compute_average_velocity(
    v_current: torch.Tensor,
    jvp: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Compute estimated average velocity for ODE trajectory estimation.

    MeanCache core formula (per paper):
        û(z_t, t, s) = v(z_t, t) + (s - t) · JVP_{r→t}

    This estimates the average velocity over interval [t, s] using the
    instantaneous velocity at t plus a linear correction based on JVP.

    NOTE: Paper uses full dt coefficient, NOT 0.5 * dt (midpoint method).
    The linear extrapolation provides better trajectory estimation for
    flow matching models.

    Args:
        v_current: Current instantaneous velocity v(z_t, t)
        jvp: Jacobian-vector product JVP_{r→t}
        dt: Timestep delta (s - t), where s is next timestep

    Returns:
        Estimated average velocity tensor û(z_t, t, s)
    """
    # Paper formula: û = v + dt * JVP (NOT 0.5 * dt)
    return v_current + dt * jvp


def compute_stability_deviation(
    v_true_avg: torch.Tensor,
    v_instant: torch.Tensor,
    jvp_k: torch.Tensor,
    dt: float
) -> float:
    """
    Compute stability deviation metric L_K(t, s) from the MeanCache paper.

    This metric measures the discrepancy between:
    - True average velocity change: u(z_t, t, s) - v(z_t, t)
    - Estimated change via JVP: (s - t) · JVP_K

    Formula: L_K(t, s) = (1/N) ||u(z_t,t,s) - v(z_t,t) - (s-t)·JVP_K||_1

    Lower values indicate that the cached JVP provides accurate trajectory
    estimation, making the step a good candidate for skipping.

    Args:
        v_true_avg: True average velocity u(z_t, t, s) - computed from actual model
        v_instant: Instantaneous velocity v(z_t, t)
        jvp_k: Cached JVP_K from K steps ago
        dt: Timestep delta (s - t)

    Returns:
        Stability deviation score (lower = more stable, better for skipping)
    """
    # True velocity change over interval
    true_diff = v_true_avg - v_instant

    # Estimated velocity change via JVP
    estimated_diff = dt * jvp_k.to(v_true_avg.device)

    # L1 deviation normalized by element count
    deviation = torch.abs(true_diff - estimated_diff).mean()
    return deviation.item()


def compute_online_L_K(
    v_new: torch.Tensor,
    v_cached: torch.Tensor,
    jvp_cached: torch.Tensor,
    dt_elapsed: float,
    eps: float = 1e-8
) -> float:
    """
    Compute online approximation of paper's L_K stability deviation.

    Measures retrospective JVP accuracy: how accurate was the cached
    JVP extrapolation at predicting the new velocity?

    This is the online equivalent of the paper's formula:
        L_K(t, s) = (1/N) ||u(z_t,t,s) - v(z_t,t) - (s-t)·JVP_K||_1

    Since we don't have the true average velocity u(z_t,t,s) online,
    we measure how well the cached JVP predicted the actual new velocity:
        L_K ≈ ||v_new - (v_cached + dt * JVP_cached)|| / ||v_new||

    Args:
        v_new: Newly computed velocity v(z_s, s)
        v_cached: Previously cached velocity v(z_t, t)
        jvp_cached: Previously cached JVP
        dt_elapsed: Time delta (sigma_cached - sigma_new)
        eps: Small value to prevent division by zero

    Returns:
        Relative prediction error (lower = better JVP accuracy, safer to skip)
    """
    # JVP-corrected prediction: what we expected v_new to be
    # NOTE: dt_elapsed is positive, but JVP correction requires addition
    predicted_v = v_cached.to(v_new.device) + dt_elapsed * jvp_cached.to(v_new.device)

    # Relative prediction error
    prediction_error = torch.abs(v_new - predicted_v).mean()
    normalizer = torch.abs(v_new).mean() + eps

    return (prediction_error / normalizer).item()


def compute_velocity_similarity(
    v_current: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    metric: str = 'l1_relative'
) -> float:
    """
    Compute similarity/distance between current and cached velocity.

    Used for quick similarity check when full stability deviation
    computation is not available (e.g., first few steps).

    Args:
        v_current: Current velocity prediction
        v_cache: Cached velocity from previous non-skipped step
        metric: Similarity metric:
            - 'l1_relative': Relative L1 distance (default)
            - 'cosine': Cosine distance (1 - cos_similarity)
            - 'l2': L2 distance

    Returns:
        Distance score (lower = more similar)
    """
    if v_cache is None:
        return float('inf')

    v_cache = v_cache.to(v_current.device)

    if metric == 'l1_relative':
        l1_distance = torch.abs(v_current - v_cache).mean()
        norm = torch.abs(v_cache).mean() + 1e-8
        return (l1_distance / norm).item()

    elif metric == 'cosine':
        v_curr_flat = v_current.flatten()
        v_cache_flat = v_cache.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            v_curr_flat.unsqueeze(0),
            v_cache_flat.unsqueeze(0)
        )
        return 1.0 - cos_sim.item()

    elif metric == 'l2':
        return torch.norm(v_current - v_cache, p=2).item()

    else:
        raise ValueError(f"Unknown metric: {metric}")


def should_skip_step(
    stability_deviation: float,
    threshold: float,
    accumulated_error: float,
    max_accumulated: float = 0.5
) -> Tuple[bool, float]:
    """
    Decide whether to skip current step based on stability deviation.

    Uses adaptive thresholding that becomes stricter as accumulated
    error increases, implementing peak suppression from PSSP.

    Args:
        stability_deviation: Current stability deviation L_K
        threshold: Base threshold for skipping
        accumulated_error: Accumulated error from previous skips
        max_accumulated: Maximum allowed accumulated error

    Returns:
        Tuple of (should_skip, new_accumulated_error)
    """
    # Adaptive threshold with peak suppression
    accumulation_factor = 1.0 - (accumulated_error / max_accumulated)
    effective_threshold = threshold * max(0.1, accumulation_factor)

    if stability_deviation < effective_threshold:
        new_accumulated = accumulated_error + stability_deviation
        if new_accumulated < max_accumulated:
            return True, new_accumulated

    # Reset accumulated error on compute
    return False, 0.0
