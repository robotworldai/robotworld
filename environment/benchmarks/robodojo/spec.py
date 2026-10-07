"""RoboProbe action contracts; no LLM client."""
from dataclasses import dataclass
import numpy as np
from .types import ActionSpace, JOINT_CHANNELS, ActionChunk

class InfrastructureFailure(RuntimeError):
    pass

@dataclass(frozen=True)
class RoboDojoActionSpec:
    """The semantics RoboDojo exposes to the joint agent policy."""

    labels: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    control_hz: float
    docs: str
    max_step: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        width = len(self.labels)
        channel_width = ActionSpace(JOINT_CHANNELS).width
        if width != channel_width:
            raise ValueError(
                f"action spec has {width} labels but the joint channels carry "
                f"{channel_width}; labels name channels by position"
            )
        if self.low.shape != (width,) or self.high.shape != (width,):
            raise ValueError("action bounds must be flat and match labels")
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("action bounds must be finite")
        if np.any(self.low > self.high):
            raise ValueError("action lower bounds must not exceed upper bounds")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if self.max_step is None:
            return
        if len(self.max_step) != width:
            raise ValueError("max_step must be one entry per action dimension")
        for entry in self.max_step:
            if entry is not None and (not np.isfinite(entry) or entry <= 0):
                raise ValueError("max_step entries must be finite and > 0 or None")

@dataclass(frozen=True)
class MotionOutcome:
    """Result of validating one model motion call."""

    chunk: ActionChunk | None
    tool_result: str
    repairable: bool = False
    notes: tuple[str, ...] = ()
