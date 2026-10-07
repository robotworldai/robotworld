# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed because we concatenate int and torch.Tensor in the type hints
from __future__ import annotations

import torch
from collections.abc import Sequence

from .circular_buffer import CircularBuffer


class DelayBuffer:
    """Delay buffer that allows retrieving stored data with delays.

    This class uses a batched circular buffer to store input data. Different to a standard circular buffer,
    which uses the LIFO (last-in-first-out) principle to retrieve the data, the delay buffer class allows
    retrieving data based on the lag set by the user. For instance, if the delay set inside the buffer
    is 1, then the second last entry from the stream is retrieved. If it is 2, then the third last entry
    and so on.

    The class supports storing a batched tensor data. This means that the shape of the appended data
    is expected to be (batch_size, ...), where the first dimension is the batch dimension. Correspondingly,
    the delay can be set separately for each batch index. If the requested delay is larger than the current
    length of the underlying buffer, the most recent entry is returned.

    .. note::
        By default, the delay buffer has no delay, meaning that the data is returned as is.
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        # set the parameters
        self._history_length = max(0, history_length)

        # the buffer size: current data plus the history length
        self._circular_buffer = CircularBuffer(self._history_length + 1, batch_size, device)

        # the minimum and maximum lags across all batch indices.
        self._min_time_lag = 0
        self._max_time_lag = 0
        # the lags for each batch index.
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._circular_buffer.batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._circular_buffer.device

    @property
    def history_length(self) -> int:
        """The history length of the delay buffer.

        If zero, only the latest data is stored. If one, the latest and the previous data are stored, and so on.
        """
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        """Minimum amount of time steps that can be delayed.

        This value cannot be negative or larger than :attr:`max_time_lag`.
        """
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum amount of time steps that can be delayed.

        This value cannot be greater than :attr:`history_length`.
        """
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag across each batch index.

        The shape of the tensor is (batch_size, ). The value at each index represents the delay for that index.
        This value is used to retrieve the data from the buffer.
        """
        return self._time_lags

    """
    Operations.
    """

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Sets the time lag for the delay buffer across the provided batch indices.

        Args:
            time_lag: The desired delay for the buffer.

              * If an integer is provided, the same delay is set for the provided batch indices.
              * If a tensor is provided, the delay is set for each batch index separately. The shape of the tensor
                should be (len(batch_ids),).

            batch_ids: The batch indices for which the time lag is set. Default is None, which sets the time lag
                for all batch indices.

        Raises:
            TypeError: If the type of the :attr:`time_lag` is not int or integer tensor.
            ValueError: If the minimum time lag is negative or the maximum time lag is larger than the history length.
        """
        # resolve batch indices
        if batch_ids is None:
            batch_ids = slice(None)

        # parse requested time_lag
        if isinstance(time_lag, int):
            # set the time lags across provided batch indices
            self._time_lags[batch_ids] = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long.")
            # set the time lags
            self._time_lags[batch_ids] = time_lag.to(device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor.")

        # compute the min and max time lag
        self._min_time_lag = int(torch.min(self._time_lags).item())
        self._max_time_lag = int(torch.max(self._time_lags).item())
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError(f"The minimum time lag cannot be negative. Received: {self._min_time_lag}")
        if self._max_time_lag > self._history_length:
            raise ValueError(
                f"The maximum time lag cannot be larger than the history length. Received: {self._max_time_lag}"
            )

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append the input data to the buffer and returns a stale version of the data based on time lag delay.

        If the requested delay is larger than the number of buffered data points since the last reset,
        the function returns the latest data. For instance, if the delay is set to 2 and only one data point
        is stored in the buffer, the function will return the latest data. If the delay is set to 2 and three
        data points are stored, the function will return the first data point.

        Args:
           data: The input data. Shape is (batch_size, ...).

        Returns:
            The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()
    

class StochasticDelayBuffer:
  """Serve stochastically delayed observations from a rolling history.

  Wraps a CircularBuffer to simulate observation delays by returning frames from T-lag
  timesteps ago, where lag is sampled from [min_lag, max_lag].

  Core Behavior
  =============

  At each timestep:
    1. Append new observation to history
    2. Sample or hold lag value (0 = no delay, 3 = 3 timesteps old)
    3. Return observation from T-lag

  Example with lag=2:
    t=0: append obs_0 → return obs_0 (not enough history)
    t=1: append obs_1 → return obs_0 (clamped to available history)
    t=2: append obs_2 → return obs_0 (lag=2, so T-2 = 0)
    t=3: append obs_3 → return obs_1 (lag=2, so T-2 = 1)

  Lag Update Policy
  =================

  Lags can be refreshed every step or periodically:

    **Every-step updates (update_period=0)**
      Each timestep may sample a new lag (subject to hold_prob).

    **Periodic updates (update_period=N)**
      Lags refresh only every N steps per environment:
        if (step_count + phase_offset) % N == 0:
            sample new lag
        else:
            keep previous lag

    **Staggered updates (per_env_phase=True)**
      Each environment gets a random phase_offset ∈ [0, N), causing
      lag updates to occur on different timesteps:
        Env 0: updates at t=0, N, 2N, ...
        Env 1: updates at t=3, N+3, 2N+3, ...
        Env 2: updates at t=7, N+7, 2N+7, ...

    **Hold probability (hold_prob=0.2)**
      Even when an update would occur, keep previous lag with 20% chance.
      Creates temporal correlation in delay patterns.

  Per-Environment vs Shared Lags
  ==============================

    **per_env=True** (default)
      Each environment has independent lag:
        Batch 0: lag=1 → returns obs from t-1
        Batch 1: lag=3 → returns obs from t-3
        Batch 2: lag=0 → returns current obs

    **per_env=False**
      All environments share one sampled lag:
        All batches: lag=2 → all return obs from t-2

  Reset Behavior
  ==============

    reset(batch_ids=[1]) clears history for specified environments:
      - Sets lag and step counter to zero
      - Clears circular buffer for those rows
      - Next append backfills their history with first new value
      - Until that append, compute() returns zeros for reset rows

  Args:
    min_lag (int, optional): Minimum lag (inclusive). Must be >= 0.
    max_lag (int, optional): Maximum lag (inclusive). Must be >= `min_lag`.
    batch_size (int, optional): Number of parallel environments (leading
      dimension of inputs).
    device (str, optional): Torch device for storage and RNG.
    per_env (bool, optional): If True, sample a separate lag per environment;
      otherwise sample one lag and share it across environments.
    hold_prob (float, optional): Probability in `[0.0, 1.0]` to keep the previous
      lag when an update would occur. Creates temporal correlation in delays.
    update_period (int, optional): If > 0, refresh lags every N steps per
      environment; if 0, consider updating every step.
    per_env_phase (bool, optional): If True and `update_period > 0`, each
      environment uses a different phase offset in `[0, update_period)`, causing
      staggered refresh steps across the batch.
    generator (torch.Generator | None, optional): Optional RNG for sampling lags.

  Examples:
    Constant delay (lag = 2):
      >>> buf = StochasticDelayBuffer(min_lag=2, max_lag=2, batch_size=4)
      >>> buf.append(obs)                # obs.shape == (4, ...)
      >>> delayed = buf.compute()        # delayed[t] = obs[t-2]

    Stochastic delay (uniform 0-3):
      >>> buf = StochasticDelayBuffer(min_lag=0, max_lag=3, batch_size=4)
      >>> buf.append(obs)
      >>> delayed = buf.compute()        # per-env lag sampled in {0,1,2,3}

    Periodic updates with staggering:
      >>> buf = StochasticDelayBuffer(
      ...     min_lag=1, max_lag=5, batch_size=8,
      ...     update_period=10,           # refresh every 10 steps
      ...     per_env_phase=True,         # stagger across envs
      ...     hold_prob=0.2               # 20% chance to hold lag
      ... )
      >>> # Env 0 refreshes at t=0,10,20,...
      >>> # Env 1 refreshes at t=3,13,23,... (random offset)
      >>> # But each refresh has 20% chance to keep previous lag
  """

  def __init__(
    self,
    min_lag: int = 0,
    max_lag: int = 3,
    batch_size: int = 1,
    device: str = "cpu",
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
    generator: torch.Generator | None = None,
  ) -> None:
    if min_lag < 0:
        raise ValueError(f"min_lag must be >= 0, got {min_lag}")
    if max_lag < min_lag:
        raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
    if not 0.0 <= hold_prob <= 1.0:
        raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
    if update_period < 0:
        raise ValueError(f"update_period must be >= 0, got {update_period}")

    self.min_lag = min_lag
    self.max_lag = max_lag
    self.batch_size = batch_size
    self.device = device
    self.per_env = per_env
    self.hold_prob = hold_prob
    self.update_period = update_period
    self.per_env_phase = per_env_phase
    self.generator = generator

    buffer_size = max_lag + 1 if max_lag > 0 else 1
    self._buffer = CircularBuffer(
        max_len=buffer_size, batch_size=batch_size, device=device
    )
    self._current_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._step_count = torch.zeros(batch_size, dtype=torch.long, device=device)

    if update_period > 0 and per_env_phase:
        self._phase_offsets = torch.randint(
            0,
            update_period,
            (batch_size,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
    else:
        self._phase_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

  @property
  def current_lags(self) -> torch.Tensor:
    """Current lag per environment. Shape: (batch_size,)."""
    return self._current_lags

  def reset(
    self, batch_ids: Sequence[int] | torch.Tensor | slice | None = None
  ) -> None:
    """Reset specified environments to initial state.

    Args:
      batch_ids: Batch indices to reset, or None to reset all.
    """
    # Convert various input types to the format CircularBuffer expects
    resolved_ids: Sequence[int] | None = None
    if isinstance(batch_ids, slice):
        indices = range(*batch_ids.indices(self.batch_size))
        resolved_ids = list(indices)
    elif isinstance(batch_ids, torch.Tensor):
        resolved_ids = batch_ids.tolist()
    elif batch_ids is not None:
        resolved_ids = batch_ids
    
    self._buffer.reset(batch_ids=resolved_ids)
    idx = slice(None) if resolved_ids is None else resolved_ids
    self._current_lags[idx] = 0
    self._step_count[idx] = 0
    if self.update_period > 0 and self.per_env_phase:
        new_phases = torch.randint(
            0,
            self.update_period,
            (self.batch_size,),
            dtype=torch.long,
            device=self.device,
            generator=self.generator,
        )
        self._phase_offsets[idx] = new_phases[idx]

  def append(self, data: torch.Tensor) -> None:
    """Append new observation to buffer.

    Args:
      data: Observation tensor of shape (batch_size, ...).
    """
    self._buffer.append(data)

  def compute(self) -> torch.Tensor:
    """Compute delayed observation for current step.

    Returns:
      Delayed observation with shape (batch_size, ...).
    """
    self.update_lags()

    # Clamp lags to valid range [0, buffer_length - 1].
    # Buffer may not be full yet (e.g., only 2 frames but sampled lag=3).
    valid_lags = torch.minimum(self._current_lags, self._buffer.current_length - 1)
    valid_lags = valid_lags.clamp_min(0)

    return self._buffer[valid_lags]

  def update_lags(self) -> None:
    """Update current lags according to configured policy."""
    if self.update_period > 0:
        phase_adjusted_count = (self._step_count + self._phase_offsets) % (
            self.update_period
        )
        should_update = phase_adjusted_count == 0
    else:
        should_update = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
    new_lags = self._sample_lags(should_update)
    self._current_lags = torch.where(should_update, new_lags, self._current_lags)
    self._step_count += 1

  def _sample_lags(self, mask: torch.Tensor) -> torch.Tensor:
    """Sample new lags for specified environments.

    Args:
      mask: Boolean mask of shape (batch_size,) indicating which envs to sample.

    Returns:
      New lags with shape (batch_size,).
    """
    if self.per_env:
        candidate_lags = torch.randint(
            self.min_lag,
            self.max_lag + 1,
            (self.batch_size,),
            dtype=torch.long,
            device=self.device,
            generator=self.generator,
        )
    else:
        shared_lag = torch.randint(
            self.min_lag,
            self.max_lag + 1,
            (1,),
            dtype=torch.long,
            device=self.device,
            generator=self.generator,
        )
        candidate_lags = shared_lag.expand(self.batch_size)

    if self.hold_prob > 0.0:
        should_sample = (
            torch.rand(
                self.batch_size,
                dtype=torch.float32,
                device=self.device,
                generator=self.generator,
            )
            >= self.hold_prob
        )
        update_mask = mask & should_sample
    else:
        update_mask = mask

    return torch.where(update_mask, candidate_lags, self._current_lags)

