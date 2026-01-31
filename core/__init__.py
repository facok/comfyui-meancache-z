"""
MeanCache core modules for Z-Image Flow Matching acceleration.

Based on UnicomAI MeanCache paper:
"From Instantaneous to Average Velocity for Accelerating Flow Matching Inference"
"""
from .meancache_state import MeanCacheState, DEFAULT_MAX_CACHE_SPAN
from .velocity_cache import (
    compute_jvp_approximation,
    compute_jvp_k,
    compute_average_velocity,
    compute_velocity_similarity,
    compute_stability_deviation,
    should_skip_step
)
from .trajectory_scheduler import TrajectoryScheduler

__all__ = [
    "MeanCacheState",
    "DEFAULT_MAX_CACHE_SPAN",
    "compute_jvp_approximation",
    "compute_jvp_k",
    "compute_average_velocity",
    "compute_velocity_similarity",
    "compute_stability_deviation",
    "should_skip_step",
    "TrajectoryScheduler"
]
