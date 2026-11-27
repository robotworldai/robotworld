"""HTTP client for communicating with a remote policy inference server.

Provides both blocking and non-blocking (async) inference requests.
See POLICY_SERVER.md for server implementation requirements.
"""

from __future__ import annotations

import base64
import io
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np


def encode_images_base64(
    images_np: list[np.ndarray], fmt: str = "jpeg", quality: int = 90
) -> list[str]:
    """Encode a batch of RGB uint8 images as base64 strings.

    Args:
        images_np: List of (H, W, 3) uint8 arrays.
        fmt: Image format ("jpeg" or "png").
        quality: JPEG quality 1-100.

    Returns:
        List of base64-encoded strings.
    """
    from PIL import Image

    encoded: list[str] = []
    for img in images_np:
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format=fmt.upper(), quality=quality)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


class PolicyClient:
    """HTTP client that talks to a policy server over REST.

    Endpoints consumed:
        GET  /info    -> model metadata (action_dim, action_horizon, …)
        POST /predict -> batched inference
        POST /reset   -> episode-reset notification (optional, best-effort)
    """

    def __init__(self, server_url: str, timeout: float = 120.0):
        import requests

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._info: dict | None = None

    # -------------------------------------------------------------- #
    # Info
    # -------------------------------------------------------------- #

    def get_info(self) -> dict:
        """Fetch model metadata from the server."""
        resp = self.session.get(f"{self.server_url}/info", timeout=self.timeout)
        resp.raise_for_status()
        self._info = resp.json()
        return self._info

    @property
    def action_horizon(self) -> int:
        if self._info is None:
            self.get_info()
        return self._info.get("action_horizon", 1)  # type: ignore[union-attr]

    @property
    def action_dim(self) -> int:
        if self._info is None:
            self.get_info()
        return self._info.get("action_dim", 8)  # type: ignore[union-attr]

    # -------------------------------------------------------------- #
    # Predict
    # -------------------------------------------------------------- #

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Blocking inference.

        Args:
            observation: Serialisable dict (see POLICY_SERVER.md for schema).

        Returns:
            ``{"actions": list | np.ndarray, "wall_latency_s": float,
               "server_latency_s": float}``.  ``actions`` is intentionally
            left in the server's JSON shape so eval.py can handle ragged
            per-env chunks instead of failing during eager ndarray conversion.
        """
        t0 = time.monotonic()
        resp = self.session.post(
            f"{self.server_url}/predict",
            json=observation,
            timeout=self.timeout,
        )
        wall = time.monotonic() - t0
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Policy server returned error: {data['error']}")
        if "actions" not in data:
            keys = ", ".join(sorted(data.keys()))
            raise RuntimeError(f"Policy server response missing 'actions' (keys: {keys})")
        return {
            "actions": data["actions"],
            "wall_latency_s": wall,
            "server_latency_s": data.get("latency_s", 0.0),
        }

    def predict_async(self, observation: dict[str, Any]) -> Future:
        """Non-blocking inference – returns a ``Future`` resolving to the
        same dict as :meth:`predict`."""
        return self._pool.submit(self.predict, observation)

    # -------------------------------------------------------------- #
    # Reset notification (optional)
    # -------------------------------------------------------------- #

    def notify_reset(self, env_ids: list[int]) -> None:
        """Best-effort reset notification; silent on failure."""
        try:
            self.session.post(
                f"{self.server_url}/reset",
                json={"env_ids": env_ids},
                timeout=5.0,
            )
        except Exception:
            pass

    # -------------------------------------------------------------- #
    # Lifecycle
    # -------------------------------------------------------------- #

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self.session.close()
