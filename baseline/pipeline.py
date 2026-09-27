"""
Unified Conventional / Baseline ATP Pipeline
"""

from __future__ import annotations

from typing import Tuple, Dict
from src.core.frame_data import FrameData
from baseline.detector import BaselineDetector
from baseline.tracker import BaselineTracker
from src.tracking.tracker_types import TrackerResult


class BaselineATP:
    """Conventional ATP coarse alignment pipeline."""

    def __init__(self):
        self.detector = BaselineDetector()
        self.tracker = BaselineTracker()

    def process_frame(self, frame_data: FrameData) -> Tuple[TrackerResult, Dict[int, Tuple[float, float]]]:
        """Run perception and tracking on a single frame."""
        det_result = self.detector.detect(frame_data)
        tracker_result = self.tracker.update(det_result)
        commands = self.tracker.compute_pan_tilt_commands()
        return tracker_result, commands
