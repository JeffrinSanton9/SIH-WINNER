"""
Conventional / Baseline Beacon Detector

Implements standard classical computer vision ATP perception:
  1. Global Otsu or Adaptive Intensity Thresholding
  2. 2D Morphological Opening/Closing to suppress single-pixel specks
  3. Connected Components analysis
  4. Intensity-weighted first-order moments (center-of-mass) centroid calculation

Known failure modes under real-world conditions:
  - High sensitivity to salt-and-pepper noise spikes (false blobs)
  - Contrast breakdown under atmospheric fog/rain (drops below threshold)
  - Severe centroid drift when airy-disk halo blurs under platform jitter
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import scipy.ndimage as ndi

from src.core.frame_data import FrameData
from src.tracking.nn1_types import Detection, NN1Result


class BaselineDetector:
    """Conventional threshold-and-moment beacon detector."""

    def __init__(
        self,
        min_blob_area: int = 4,
        max_blob_area: int = 1500,
        fixed_threshold_floor: int = 35,
        max_detections: int = 5,
    ):
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.fixed_threshold_floor = fixed_threshold_floor
        self.max_detections = max_detections

    def _compute_otsu_threshold(self, image: np.ndarray) -> int:
        """Compute Otsu's optimal threshold on grayscale image."""
        hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
        total_pixels = image.size

        current_max = 0.0
        threshold = self.fixed_threshold_floor
        sum_total = np.dot(np.arange(256), hist)

        weight_bg = 0
        sum_bg = 0

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0:
                break

            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            # Between-class variance
            var_between = float(weight_bg) * float(weight_fg) * ((mean_bg - mean_fg) ** 2)

            if var_between > current_max:
                current_max = var_between
                threshold = t

        # Never drop below minimum noise floor
        return max(threshold, self.fixed_threshold_floor)

    def detect(self, frame_data: FrameData) -> NN1Result:
        """Detect beacon centroids from raw monochrome frame."""
        image = frame_data.image
        if image is None:
            return NN1Result(
                frame_index=frame_data.frame_index,
                timestamp=frame_data.timestamp,
                detections=[],
                inference_time_ms=0.0,
                detector_type="conventional_otsu",
            )

        import time
        t0 = time.perf_counter()

        # 1. Otsu thresholding
        thresh_val = self._compute_otsu_threshold(image)
        binary_mask = image >= thresh_val

        # 2. Morphological 3x3 structuring element opening to clear small noise
        struct = ndi.generate_binary_structure(2, 1)  # 4-connectivity
        cleaned_mask = ndi.binary_opening(binary_mask, structure=struct)

        # 3. Connected components labeling
        labeled, num_features = ndi.label(cleaned_mask)

        detections: List[Detection] = []
        if num_features > 0:
            # Measure component sizes
            component_sizes = ndi.sum(cleaned_mask, labeled, range(1, num_features + 1))
            if np.isscalar(component_sizes):
                component_sizes = [component_sizes]

            valid_indices = [
                i + 1 for i, size in enumerate(component_sizes)
                if self.min_blob_area <= size <= self.max_blob_area
            ]

            if valid_indices:
                # Fast vectorized center of mass and peak intensity in C
                coms = ndi.center_of_mass(image, labeled, valid_indices)
                max_vals = ndi.maximum(image, labeled, valid_indices)

                if not isinstance(coms, list):
                    coms = [coms]
                if np.isscalar(max_vals):
                    max_vals = [max_vals]

                for (cy, cx), max_val in zip(coms, max_vals):
                    if not (math.isnan(cy) or math.isnan(cx)):
                        max_intensity = float(max_val)
                        confidence = min(1.0, max_intensity / 255.0)
                        detections.append(Detection(
                            x=float(cx),
                            y=float(cy),
                            confidence=confidence,
                            peak_intensity=int(max_intensity),
                        ))

        # Sort by intensity and limit to max_detections
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[:self.max_detections]

        return NN1Result(
            frame_index=frame_data.frame_index,
            timestamp=frame_data.timestamp,
            detections=detections,
            disturbance_descriptor=frame_data.disturbance_descriptor,
            camera_pan_deg=frame_data.camera_pan_deg,
            camera_tilt_deg=frame_data.camera_tilt_deg,
            capture_rate_hz=frame_data.capture_rate_hz,
        )
