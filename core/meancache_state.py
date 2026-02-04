"""
MeanCache state management for tracking velocity cache across sampling steps.

Based on UnicomAI MeanCache: https://unicomai.github.io/MeanCache/

Key concepts:
- v_history: Stores recent K velocities for JVP_K computation
- t_history: Corresponding timesteps (sigma values)
- Cache span K: JVP computed over K steps for stability
"""
from typing import Optional, Dict, Any, List
import torch


# Default max cache span for JVP_K computation
DEFAULT_MAX_CACHE_SPAN = 3


class MeanCacheState:
    """
    Manages state across sampling steps for MeanCache algorithm.

    State is passed via transformer_options and persists across steps.
    Supports multiple prediction states for CFG (conditional/unconditional).
    """

    def __init__(self, cache_device: str = 'cpu', max_cache_span: int = DEFAULT_MAX_CACHE_SPAN):
        """
        Initialize MeanCache state manager.

        Args:
            cache_device: Device for storing cached velocities ('cpu' or 'cuda')
            max_cache_span: Maximum cache span K for JVP_K computation (default 3)
        """
        self.cache_device = cache_device
        self.max_cache_span = max_cache_span
        self.states: Dict[int, Dict[str, Any]] = {}
        self._next_pred_id: int = 0
        self.last_sigma: Optional[float] = None  # For split CFG detection
        self.split_cfg_unified_error: Optional[float] = None  # Unified error for consistent CFG decisions
        self.scheduler: Optional[Any] = None  # PSSP scheduler for skip decisions

    def new_prediction(self, cache_device: Optional[str] = None) -> int:
        """
        Create new prediction state and return its ID for CFG tracking.

        Args:
            cache_device: Optional override for cache device

        Returns:
            Prediction ID (0 for conditional, 1 for unconditional)
        """
        if cache_device is not None:
            self.cache_device = cache_device

        pred_id = self._next_pred_id
        self._next_pred_id += 1

        self.states[pred_id] = {
            # Velocity cache for JVP computation
            'v_prev': None,              # Previous velocity v_{t-1}
            'v_cache': None,             # Cached velocity for skip reuse
            't_prev': None,              # Previous timestep sigma_{t-1}
            'dt_prev': None,             # Previous dt for JVP approximation

            # JVP cache for skip correction (paper core algorithm)
            'jvp_cache': None,           # Cached JVP for velocity correction during skip
            'sigma_cache': None,         # Sigma at which v_cache/jvp_cache were computed

            # Multi-step history for JVP_K computation (paper Section 3.2)
            'v_history': [],             # Recent K velocities [v_{t-K}, ..., v_t]
            't_history': [],             # Corresponding timesteps [sigma_{t-K}, ..., sigma_t]

            # Skip tracking and metrics
            'accumulated_distance': 0.0, # Accumulated L1 distance since last compute
            'accumulated_error': 0.0,    # Accumulated error for adaptive thresholding
            'skipped_steps': [],         # List of step indices that were skipped
            'step_index': 0,             # Current step counter

            # Trajectory optimization state
            'trajectory_budget': None,   # Remaining compute budget from scheduler
            'scheduled_skip_mask': None, # Pre-computed skip mask from PSSP
        }
        return pred_id

    def update(self, pred_id: int, **kwargs) -> None:
        """
        Update state for specific prediction (cond/uncond).

        Args:
            pred_id: Prediction ID to update
            **kwargs: State fields to update
        """
        if pred_id not in self.states:
            return
        for key, value in kwargs.items():
            if key in self.states[pred_id]:
                self.states[pred_id][key] = value

    def get(self, pred_id: int) -> Dict[str, Any]:
        """
        Retrieve state for specific prediction.

        Args:
            pred_id: Prediction ID

        Returns:
            State dictionary or empty dict if not found
        """
        return self.states.get(pred_id, {})

    def get_or_create(self, pred_id: int, cache_device: Optional[str] = None) -> Dict[str, Any]:
        """
        Get existing state or create new one if it doesn't exist.

        Args:
            pred_id: Prediction ID
            cache_device: Optional device for new state

        Returns:
            State dictionary
        """
        if pred_id not in self.states:
            # Create states up to and including pred_id
            while self._next_pred_id <= pred_id:
                self.new_prediction(cache_device)
        return self.states.get(pred_id, {})

    def update_history(
        self,
        pred_id: int,
        velocity: torch.Tensor,
        timestep: float
    ) -> None:
        """
        Update velocity and timestep history for JVP_K computation.

        Maintains a sliding window of recent velocities and timesteps,
        limited by max_cache_span.

        Args:
            pred_id: Prediction ID
            velocity: Current velocity tensor (will be moved to cache_device)
            timestep: Current timestep (sigma value)
        """
        if pred_id not in self.states:
            return

        state = self.states[pred_id]

        # Store velocity on cache device to save GPU memory
        v_cached = velocity.detach().clone().to(self.cache_device)
        state['v_history'].append(v_cached)
        state['t_history'].append(timestep)

        # Maintain sliding window of size max_cache_span + 1
        max_len = self.max_cache_span + 1
        if len(state['v_history']) > max_len:
            # Remove oldest entry and free memory
            old_v = state['v_history'].pop(0)
            del old_v
            state['t_history'].pop(0)

    def get_jvp_k(
        self,
        pred_id: int,
        k: int,
        eps: float = 1e-8
    ) -> Optional[torch.Tensor]:
        """
        Compute JVP with cache span K from stored history.

        JVP_K = (v_now - v_{now-K}) / (t_{now-K} - t_now)

        Args:
            pred_id: Prediction ID
            k: Cache span (number of steps to look back)
            eps: Small value to prevent division by zero

        Returns:
            JVP_K tensor or None if insufficient history
        """
        if pred_id not in self.states:
            return None

        state = self.states[pred_id]
        v_history = state['v_history']
        t_history = state['t_history']

        if len(v_history) < k + 1 or len(t_history) < k + 1:
            return None

        v_now = v_history[-1]
        v_k_ago = v_history[-(k + 1)]
        t_now = t_history[-1]
        t_k_ago = t_history[-(k + 1)]

        # sigma decreases during sampling, so t_k_ago > t_now
        dt = t_k_ago - t_now
        if abs(dt) < eps:
            return None

        return (v_now - v_k_ago.to(v_now.device)) / dt

    def get_history_length(self, pred_id: int) -> int:
        """
        Get current history length for a prediction.

        Args:
            pred_id: Prediction ID

        Returns:
            Number of stored history entries
        """
        if pred_id not in self.states:
            return 0
        return len(self.states[pred_id].get('v_history', []))

    def clear_all(self) -> None:
        """Reset all states between sampling runs."""
        # Clean up tensors to free memory
        for state in self.states.values():
            if state.get('v_prev') is not None:
                del state['v_prev']
            if state.get('v_cache') is not None:
                del state['v_cache']
            # Clean up history tensors
            for v in state.get('v_history', []):
                del v
            state['v_history'] = []
            state['t_history'] = []

        self.states = {}
        self._next_pred_id = 0
        self.last_sigma = None
        self.split_cfg_unified_error = None
        self.scheduler = None

    def increment_step(self, pred_id: int) -> None:
        """
        Increment step counter for a prediction.

        Args:
            pred_id: Prediction ID
        """
        if pred_id in self.states:
            self.states[pred_id]['step_index'] += 1

    def record_skip(self, pred_id: int, step_index: int) -> None:
        """
        Record that a step was skipped.

        Args:
            pred_id: Prediction ID
            step_index: Index of skipped step
        """
        if pred_id in self.states:
            self.states[pred_id]['skipped_steps'].append(step_index)

    def get_skip_count(self, pred_id: int) -> int:
        """
        Get number of skipped steps for a prediction.

        Args:
            pred_id: Prediction ID

        Returns:
            Number of skipped steps
        """
        if pred_id in self.states:
            return len(self.states[pred_id]['skipped_steps'])
        return 0

    def get_total_skip_count(self) -> int:
        """
        Get total number of skipped steps across all predictions.

        Returns:
            Total skipped steps
        """
        total = 0
        for state in self.states.values():
            total += len(state.get('skipped_steps', []))
        return total

    def get_report(self) -> Dict[str, Any]:
        """
        Get summary report of MeanCache performance.

        Returns:
            Dictionary with skip statistics per prediction
        """
        report = {}
        state_names = {0: "conditional", 1: "unconditional"}

        for pred_id, state in self.states.items():
            name = state_names.get(pred_id, f"prediction_{pred_id}")
            report[name] = {
                'skipped_steps': state.get('skipped_steps', []),
                'skip_count': len(state.get('skipped_steps', [])),
                'total_steps': state.get('step_index', 0),
            }

        return report
