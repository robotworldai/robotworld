# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive sampling utilities for run_eval.py.

Decides whether to run another batch of episodes for a task, based on the width
of the Bayesian credible interval over the success rate. The estimator is the
Beta posterior with uniform prior:  p ~ Beta(k+1, n-k+1).

Defaults follow the TRI LBM sim protocol (Toyota Research Institute, "A Careful
Examination of Large Behavior Models for Multitask Dexterous Manipulation",
arXiv:2507.05331), which uses 200 rollouts per task in sim. A 95% CI width of
~0.14 is the worst-case (k/n=0.5) width at n=200, so target_width=0.14 with
n_max=200 reproduces their effective precision while letting easy tasks stop
earlier. Tighten to 0.10 for publication-grade precision; loosen to ~0.27 to
match their 50-rollout real-world protocol.
"""

from scipy.stats import beta


def should_continue_sampling(
    k: int,
    n: int,
    target_width: float = 0.14,
    n_min: int = 10,
    n_max: int = 200,
) -> bool:
    """Return True if the eval should run another batch on this task.

    Stopping rule depends only on uncertainty (CI width), not on the value of
    the estimate, so the resulting estimator stays unbiased.
    """
    if n >= n_max:
        return False
    if n < n_min:
        return True
    lo, hi = beta.ppf([0.025, 0.975], k + 1, n - k + 1)
    return (hi - lo) > target_width


def count_task_episodes(episode_results: list[dict], env_name: str) -> tuple[int, int]:
    """Count (successes, total) for one task from the running episode_results list."""
    eps = [ep for ep in episode_results if ep.get("env_name") == env_name]
    n = len(eps)
    k = sum(1 for ep in eps if ep.get("success"))
    return k, n
