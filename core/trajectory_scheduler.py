"""
Trajectory Stability Scheduling for MeanCache.

Implements Peak-Suppressed Shortest Path (PSSP) algorithm for
optimal compute budget allocation across sampling steps.

Paper formula:
    π* = argmin_{π ∈ P(T,0)} Σ_{e ∈ π} C(e)^γ   s.t. |π| ≤ B

Where:
- P(T,0): Set of all paths from timestep T to 0
- C(e): Cost (deviation estimate) of edge e
- γ: Penalty exponent for peak suppression (suppresses large deviations)
- B: Compute budget (max number of steps)

Reference: UnicomAI MeanCache paper
"""
import torch
from typing import List, Tuple, Optional


class TrajectoryScheduler:
    """
    Implements Peak-Suppressed Shortest Path (PSSP) algorithm using
    dynamic programming for optimal compute budget allocation.

    Models the sampling process as a graph shortest path problem:
    - Nodes: timesteps t_0, t_1, ..., t_N
    - Edges: transitions between timesteps (compute or skip)
    - Edge weights: C(e)^γ where C(e) is deviation estimate

    Peak suppression (γ > 1) penalizes large deviations more heavily,
    preferring multiple small skips over one large skip.
    """

    def __init__(
        self,
        total_steps: int,
        skip_budget: float = 0.3,
        gamma: float = 2.0,
        peak_threshold: float = 0.15,
        min_compute_steps: int = 4,
        critical_start_ratio: float = None,
        critical_end_ratio: float = None
    ):
        """
        Initialize the trajectory scheduler.

        Args:
            total_steps: Total number of sampling steps
            skip_budget: Fraction of steps that can be skipped (0.0-0.5)
            gamma: Peak suppression exponent (higher = penalize large skips more)
            peak_threshold: Max allowed single-step deviation before forced compute
            min_compute_steps: Minimum steps that must be computed
            critical_start_ratio: Fraction of early steps to protect (default 20%)
            critical_end_ratio: Fraction where late critical zone begins (default 80%)
        """
        self.total_steps = max(1, total_steps)
        self.skip_budget = max(0.0, min(0.75, skip_budget))
        self.gamma = gamma
        self.peak_threshold = peak_threshold
        self.min_compute_steps = min_compute_steps
        # Adjust critical zone based on skip budget: higher budget = smaller protected zones
        if critical_start_ratio is None:
            # Quality (0.2) -> Turbo (0.05)
            self.critical_start_ratio = max(0.05, 0.20 - self.skip_budget * 0.25)
        else:
            self.critical_start_ratio = critical_start_ratio
        if critical_end_ratio is None:
            # Quality (0.8) -> Turbo (0.95)
            self.critical_end_ratio = min(0.95, 0.80 + self.skip_budget * 0.20)
        else:
            self.critical_end_ratio = critical_end_ratio

        # Will be computed when deviation estimates are available
        self.skip_mask: List[bool] = [False] * total_steps
        self.step_weights: List[float] = [1.0] * total_steps

        # Initialize with heuristic schedule (no deviation data yet)
        self._compute_heuristic_schedule()

    def _compute_heuristic_schedule(self) -> None:
        """
        Compute initial heuristic schedule when no deviation estimates available.

        Protects critical early/late steps and distributes skips evenly
        in the middle zone with minimum spacing.
        """
        n = self.total_steps
        max_skips = max(0, int(n * self.skip_budget))

        # Compute critical zone boundaries
        critical_start = max(1, int(n * self.critical_start_ratio))
        critical_end = min(n - 1, int(n * self.critical_end_ratio))

        # Initialize all as compute (False = don't skip)
        self.skip_mask = [False] * n
        self.step_weights = [1.0] * n

        # No skips if budget is 0 or too few steps
        if max_skips == 0 or n <= self.min_compute_steps:
            return

        # Identify skip candidates in middle zone
        skip_candidates = list(range(critical_start, critical_end))

        if len(skip_candidates) == 0:
            return

        # Calculate minimum spacing between skips
        # Allow spacing of 1 for high skip budgets to achieve aggressive skipping
        min_spacing = max(1, len(skip_candidates) // (max_skips + 1))

        # Assign skips with even spacing
        skips_assigned = 0
        for i in range(0, len(skip_candidates), min_spacing):
            if skips_assigned >= max_skips:
                break
            idx = skip_candidates[i]
            self.skip_mask[idx] = True
            skips_assigned += 1

        # Compute step weights
        self._update_step_weights()

    def _update_step_weights(self) -> None:
        """Update importance weights based on skip mask and critical zones."""
        n = self.total_steps
        self.step_weights = [1.0] * n

        for i in range(n):
            # Base weight from skip mask
            if i < len(self.skip_mask) and self.skip_mask[i]:
                self.step_weights[i] = 0.5

            # Boost early steps
            if i < n * self.critical_start_ratio:
                self.step_weights[i] *= 1.5

            # Boost final steps
            if i >= n * self.critical_end_ratio:
                self.step_weights[i] *= 1.3

    def compute_optimal_schedule_pssp(
        self,
        deviation_estimates: List[float],
        max_consecutive_skips: int = 3,
        protect_steps: Optional[List[int]] = None
    ) -> List[bool]:
        """
        Compute optimal skip schedule using PSSP dynamic programming.

        Paper formula: π* = argmin Σ C(e)^γ s.t. |π| ≤ B

        DP formulation:
        - State: dp[i][j] = min cost to reach step i having skipped j steps
        - Transition: either compute step i (cost 0) or skip (cost C(e)^γ)

        Args:
            deviation_estimates: List of deviation estimates for each step
            max_consecutive_skips: Maximum consecutive steps that can be skipped
            protect_steps: Specific step indices to never skip. If None, only
                           protects steps with no calibration data (sample_count=0).

        Returns:
            List of booleans where True = skip this step
        """
        n = self.total_steps
        max_skips = max(0, int(n * self.skip_budget))

        # With calibration data, let the data drive decisions via gamma penalty.
        # Only protect steps that have no data or are explicitly specified.
        if protect_steps is not None:
            protected_set = set(protect_steps)
        else:
            # Default: protect step 0 (first step, no L_K) and last step
            protected_set = {0, n - 1}

        # Handle edge cases
        if max_skips == 0 or n <= self.min_compute_steps:
            self.skip_mask = [False] * n
            return self.skip_mask

        if len(deviation_estimates) != n:
            # Fall back to heuristic if wrong size
            self._compute_heuristic_schedule()
            return self.skip_mask

        # Initialize DP tables
        INF = float('inf')
        # dp[i][j] = minimum cost to reach step i with j total skips
        dp = [[INF] * (max_skips + 1) for _ in range(n + 1)]
        # parent[i][j] = (prev_step, prev_skips, was_skip) for backtracking
        parent = [[None] * (max_skips + 1) for _ in range(n + 1)]
        # consecutive[i][j] = number of consecutive skips ending at step i with j total skips
        consecutive = [[0] * (max_skips + 1) for _ in range(n + 1)]

        # Base case: start at step 0 with 0 skips, 0 cost
        dp[0][0] = 0

        # Fill DP table
        for i in range(1, n + 1):
            step_idx = i - 1  # 0-indexed step
            is_protected = step_idx in protected_set

            for j in range(min(i, max_skips) + 1):
                # Option 1: Compute this step (don't skip)
                if dp[i - 1][j] < dp[i][j]:
                    dp[i][j] = dp[i - 1][j]
                    parent[i][j] = (i - 1, j, False)
                    consecutive[i][j] = 0

                # Option 2: Skip this step
                if j > 0 and not is_protected:
                    prev_consecutive = consecutive[i - 1][j - 1]

                    if prev_consecutive < max_consecutive_skips:
                        deviation = deviation_estimates[step_idx]
                        skip_cost = (deviation ** self.gamma) if deviation > 0 else 0

                        new_cost = dp[i - 1][j - 1] + skip_cost

                        if new_cost < dp[i][j]:
                            dp[i][j] = new_cost
                            parent[i][j] = (i - 1, j - 1, True)
                            consecutive[i][j] = prev_consecutive + 1

        # Find best final state: USE full skip budget for maximum speedup.
        # The DP minimizes deviation cost for a GIVEN number of skips.
        # We want to use as many skips as the budget allows.
        best_j = 0
        best_cost = dp[n][0]

        # Try full budget first, fall back to fewer if not achievable
        for j in range(max_skips, -1, -1):
            if dp[n][j] < INF:
                best_j = j
                best_cost = dp[n][j]
                break

        # Backtrack to find skip mask
        skip_mask = [False] * n
        current_i, current_j = n, best_j

        while current_i > 0 and parent[current_i][current_j] is not None:
            prev_i, prev_j, was_skip = parent[current_i][current_j]
            if was_skip:
                skip_mask[current_i - 1] = True  # Convert to 0-indexed
            current_i, current_j = prev_i, prev_j

        self.skip_mask = skip_mask
        self._update_step_weights()
        return skip_mask

    def get_skip_decision(
        self,
        step_index: int,
        velocity_similarity: float,
        accumulated_error: float
    ) -> Tuple[bool, float]:
        """
        Get skip decision for current step using PSSP schedule.

        Combines pre-computed schedule with runtime velocity analysis
        to make adaptive decisions.

        Args:
            step_index: Current step index (0-based)
            velocity_similarity: Measured velocity similarity/distance
            accumulated_error: Accumulated trajectory error from previous skips

        Returns:
            Tuple of (should_skip, new_accumulated_error)
            - should_skip: True if step can be safely skipped
            - new_accumulated_error: Updated accumulated error
        """
        # Out of bounds check - fall back to deviation-based decision
        if step_index < 0 or step_index >= len(self.skip_mask):
            # Use opportunistic skip logic for out-of-bounds steps
            max_accumulated = 0.5
            if velocity_similarity < self.peak_threshold * 0.5:
                if accumulated_error < max_accumulated * 0.6:
                    new_error = accumulated_error + velocity_similarity
                    return True, new_error
            return False, 0.0

        scheduled_skip = self.skip_mask[step_index]

        # Override: don't skip if velocity changed too much (peak suppression)
        if scheduled_skip and velocity_similarity > self.peak_threshold:
            return False, 0.0

        # Override: don't skip if too much error accumulated
        # Scale max_accumulated based on skip_budget to allow higher skipping for aggressive modes
        max_accumulated = 0.5 + self.skip_budget * 0.5  # 0.5 -> 0.875 for high budgets
        if scheduled_skip and accumulated_error > max_accumulated * 0.8:
            return False, 0.0

        # Opportunistic skip if very stable (not scheduled but safe)
        if not scheduled_skip:
            if velocity_similarity < self.peak_threshold * 0.3:
                if accumulated_error < max_accumulated * 0.4:
                    # Allow opportunistic skip
                    new_error = accumulated_error + velocity_similarity
                    return True, new_error

        # Follow schedule
        if scheduled_skip:
            new_error = accumulated_error + velocity_similarity
            return True, new_error

        # Compute step - reset accumulated error
        return False, 0.0

    def get_timestep_weight(self, step_index: int) -> float:
        """
        Get importance weight for a specific timestep.

        Args:
            step_index: Step index

        Returns:
            Weight value (higher = more important)
        """
        if 0 <= step_index < len(self.step_weights):
            return self.step_weights[step_index]
        return 1.0

    def get_schedule_summary(self) -> dict:
        """
        Get summary of the skip schedule.

        Returns:
            Dictionary with schedule statistics
        """
        skip_indices = [i for i, skip in enumerate(self.skip_mask) if skip]
        return {
            'total_steps': self.total_steps,
            'scheduled_skips': len(skip_indices),
            'skip_ratio': len(skip_indices) / max(1, self.total_steps),
            'skip_indices': skip_indices,
            'gamma': self.gamma,
            'critical_start': int(self.total_steps * self.critical_start_ratio),
            'critical_end': int(self.total_steps * self.critical_end_ratio),
        }

    def adjust_for_sigmas(self, sigmas: torch.Tensor) -> None:
        """
        Adjust schedule based on actual sigma values.

        Large sigma transitions are more critical and should not be skipped.

        Args:
            sigmas: Tensor of sigma values for each step
        """
        if sigmas is None or len(sigmas) <= 1:
            return

        n = len(sigmas) - 1  # Number of steps

        if n != self.total_steps:
            # Recompute if step count changed
            self.total_steps = n
            self._compute_heuristic_schedule()
            return

        # Identify critical sigma transitions
        for i in range(n):
            sigma_ratio = sigmas[i].item() / max(sigmas[i + 1].item(), 1e-8)

            # Large sigma change = critical step, don't skip
            if sigma_ratio > 2.0 and i < len(self.skip_mask):
                self.skip_mask[i] = False
                self.step_weights[i] *= 1.2
