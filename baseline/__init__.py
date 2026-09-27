"""
Baseline / Conventional Approach for Coarse Alignment of Mobile FSOC Terminals.

This package implements the classical avionics/computer-vision ATP stack:
  - Otsu / Adaptive Thresholding + Morphological filtering + Intensity-weighted Centroids
  - Greedy Nearest Neighbor (GNN) Data Association
  - Standard Linear Constant-Velocity Kalman Filter (No Recurrent GRU)
  - Reactive Proportional Gimbal Control with Static 30 Hz Sampling (No Dynamic Loss Ramping)
"""

from baseline.detector import BaselineDetector
from baseline.tracker import BaselineTracker
from baseline.pipeline import BaselineATP

__all__ = ["BaselineDetector", "BaselineTracker", "BaselineATP"]
