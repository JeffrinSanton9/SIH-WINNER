"""
SADHA — Video Player & Stream Ingestion Pipeline
Handles frame-by-frame decoding, looping, scrubbing, and preprocessing
for external mission video telemetry (.mp4, .avi, .mkv).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoPlayer:
    """Frame-by-frame video stream decoder for Benchmark-2 evaluation."""

    def __init__(self, video_path: str, loop: bool = True):
        self.path = os.path.abspath(video_path)
        self.filename = os.path.basename(video_path)
        self.loop = loop

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {self.path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            # Fallback if frame count cannot be read
            self.total_frames = 1

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if (fps and fps > 0.0) else 30.0

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        self.current_frame_idx = 0

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Read the next video frame converted to monochrome uint8 (H, W)."""
        if not self.is_opened:
            return False, None, self.current_frame_idx

        ret, frame = self.cap.read()
        if not ret:
            if self.loop and self.total_frames > 0:
                self.seek(0)
                ret, frame = self.cap.read()
            if not ret:
                return False, None, self.current_frame_idx

        # Accurate frame index tracking
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.current_frame_idx = max(0, pos)

        # Convert to monochrome uint8
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                gray = frame[:, :, 0]
        else:
            gray = frame

        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        return True, gray, self.current_frame_idx

    def seek(self, frame_idx: int) -> Tuple[bool, Optional[np.ndarray], int]:
        """Seek to a specific frame index and return that frame."""
        if not self.is_opened:
            return False, None, self.current_frame_idx

        target = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        return self.read_frame()

    def restart(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Rewind video to frame 0."""
        return self.seek(0)

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
