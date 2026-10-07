# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging

import imageio.v2 as imageio
import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """Streaming H.264 video writer.

    Pipes RGB frames to an ffmpeg subprocess (via imageio-ffmpeg) which encodes
    libx264/yuv420p with +faststart so the output plays in Chrome/Safari/Firefox.
    Memory is bounded by the encoder's lookahead, not by video length.
    """

    def __init__(self, video_path: str, fps: int):
        """Initialize video writer.

        Args:
            video_path: Path to output video file
            fps: Frames per second
        """
        self.video_path = video_path
        self.fps = fps
        self._writer = None
        atexit.register(self.release)

    def write(self, frame: np.ndarray):
        if frame is None:
            print(f"No frame to write to video writer; nothing written to file '{self.video_path}'")
            return

        if self._writer is None:
            # macro_block_size=2 pads odd-dimensioned frames so libx264+yuv420p (even-only) is safe.
            # ultrafast/threads=1 keeps per-encoder CPU low so num_envs=10 doesn't oversubscribe.
            self._writer = imageio.get_writer(
                self.video_path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=2,
                ffmpeg_params=[
                    "-preset", "ultrafast",
                    "-threads", "1",
                    "-crf", "23",
                    "-movflags", "+faststart",
                ],
            )

        self._writer.append_data(frame)

    def release(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("Failed to close video writer for '%s'", self.video_path)
            self._writer = None

    def __del__(self):
        self.release()
