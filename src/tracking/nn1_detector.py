"""
SADHA — NN1 Beacon Detector

High-level detector wrapping:
  1. A learned CNN heatmap model (``BeaconHeatmapNet``) — primary path.
  2. A classical image-processing fallback (adaptive threshold + connected
     components + weighted-moment centroid) — used when no trained model
     weights are available or as a baseline for comparison.

Both paths produce identical ``NN1Result`` records, so every downstream
consumer (NN2, Kalman, Hungarian) is agnostic to which ran.

Usage::

    detector = NN1Detector()              # auto-selects CNN or classical
    detector = NN1Detector(mode="cnn")    # force CNN (requires weights)
    detector = NN1Detector(mode="classical")

    result: NN1Result = detector.detect(frame_data)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.core.frame_data import FrameData
from src.tracking.nn1_types import Detection, NN1Result

# PyTorch is a soft dependency — classical mode works without it
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from src.tracking.nn1_model import BeaconHeatmapNet, create_model
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    # ImportError: torch not installed
    # OSError: DLL load failure on Windows (Python 3.14 / missing VC++ runtime)
    pass


# Default path for saved model weights
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "nn1_weights.pth"


class NN1Detector:
    """Frame-level beacon detector — runs on every frame, unconditionally.

    Parameters
    ----------
    mode : str
        ``"auto"`` (default) — CNN if weights exist, else classical.
        ``"cnn"`` — force CNN (raises if no weights and no training).
        ``"classical"`` — force classical pipeline.
    weights_path : Path or None
        Override default weights file location.
    detection_threshold : float
        Minimum confidence score to emit a detection (0–1).
    max_detections : int
        Maximum detections per frame (caps output list length).
    """

    def __init__(
        self,
        mode: str = "auto",
        weights_path: Optional[Path] = None,
        detection_threshold: float = 0.35,
        max_detections: int = 5,
    ):
        self.mode = mode
        self.detection_threshold = detection_threshold
        self.max_detections = max_detections
        self._weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS

        # CNN state
        self._model: Optional[object] = None
        self._device = None
        self._use_cnn = False

        # Classical detector tuning
        self._classical_threshold = 35   # Intensity floor for binary mask
        self._min_blob_area = 4          # Reject sub-4px noise spikes
        self._max_blob_area = 2000       # Reject massive artefacts

        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------
    def _init_backend(self):
        if self.mode == "classical":
            self._use_cnn = False
            return

        if not _TORCH_AVAILABLE:
            if self.mode == "cnn":
                raise RuntimeError(
                    "NN1 mode='cnn' requested but PyTorch is not installed."
                )
            self._use_cnn = False
            return

        # Try to load pre-trained weights
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._weights_path.exists():
            self._model = create_model(self._device)
            state = torch.load(
                self._weights_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state)
            self._model.eval()
            self._use_cnn = True
        elif self.mode == "cnn":
            # No weights but CNN was forced — create untrained model
            # (caller must train before meaningful use)
            self._model = create_model(self._device)
            self._model.eval()
            self._use_cnn = True
        else:
            # auto mode, no weights → classical fallback
            self._use_cnn = False

    def set_model(self, model) -> None:
        """Hot-swap a trained model (e.g. after training completes)."""
        self._model = model
        self._model.eval()
        self._use_cnn = True

    # ------------------------------------------------------------------
    # Primary detection interface (Interface Boundary: Module 1 → NN1 → Module 2.2)
    # ------------------------------------------------------------------
    def detect(self, frame_data: FrameData) -> NN1Result:
        """Detect optical beacon candidate centroids in a single sensor frame.

        Parameters
        ----------
        frame_data : FrameData
            Input frame packet containing 640x480 monochrome uint8 image,
            frame sequence index, timestamp, and optional disturbance telemetry.

        Returns
        -------
        NN1Result
            Immutable output record containing:
            - `frame_index`: Matched to input frame_data.
            - `timestamp`: Matched to input frame_data.
            - `detections`: List of 0-to-N sub-pixel `Detection` objects, sorted
              by descending confidence and filtered by `detection_threshold`.
            - `disturbance_descriptor`: Forwarded 8-D environmental vector.
            - `camera_pan_deg`, `camera_tilt_deg`, `capture_rate_hz`: Telemetry pass-through.

        Interface Boundary
        ------------------
        Consumes the output of Module 1 (or external video player) and emits
        the foundational perception record consumed by the Hungarian association
        and Kalman filtering layers in Module 2.2.
        """
        image = frame_data.image  # (H, W) uint8 monochrome

        if self._use_cnn and self._model is not None:
            detections = self._detect_cnn(image)
        else:
            detections = self._detect_classical(image)

        # Apply confidence threshold and cap
        detections = [d for d in detections if d.confidence >= self.detection_threshold]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[:self.max_detections]

        return NN1Result(
            frame_index=frame_data.frame_index,
            timestamp=frame_data.timestamp,
            detections=detections,
            disturbance_descriptor=(
                list(frame_data.disturbance_descriptor)
                if frame_data.disturbance_descriptor else None
            ),
            camera_pan_deg=frame_data.camera_pan_deg,
            camera_tilt_deg=frame_data.camera_tilt_deg,
            capture_rate_hz=frame_data.capture_rate_hz,
        )

    # ------------------------------------------------------------------
    # CNN Detection Path
    # ------------------------------------------------------------------
    def _detect_cnn(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]

        # Normalise and convert to tensor
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, H, W)

        with torch.no_grad():
            heatmap = self._model(tensor)  # (1, 1, H/4, W/4)

        heatmap = heatmap.squeeze(0).squeeze(0)  # (H/4, W/4)
        hm_h, hm_w = heatmap.shape

        # Scale factor from heatmap space to full image space
        scale_x = w / hm_w
        scale_y = h / hm_h

        # Non-maximum suppression via max-pooling
        # Kernel must be wider than the beacon blob at quarter resolution
        # (10px beacon → ~3px at 1/4 scale, so kernel 11 covers ±5)
        nms_kernel = 11
        heatmap_padded = heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,H/4,W/4)
        pooled = F.max_pool2d(
            heatmap_padded, kernel_size=nms_kernel, stride=1,
            padding=nms_kernel // 2
        )
        pooled = pooled.squeeze(0).squeeze(0)

        # Peaks: points where heatmap equals pooled (local maxima) and above threshold
        peak_mask = (heatmap == pooled) & (heatmap > self.detection_threshold)
        peak_coords = torch.nonzero(peak_mask, as_tuple=False)  # (N, 2) — [row, col]

        # Collect raw peaks sorted by confidence
        raw_peaks = []
        for i in range(peak_coords.shape[0]):
            row = peak_coords[i, 0].item()
            col = peak_coords[i, 1].item()
            conf = heatmap[row, col].item()
            raw_peaks.append((row, col, conf))
        raw_peaks.sort(key=lambda p: p[2], reverse=True)

        # Spatial de-duplication: suppress peaks within min_dist of a stronger peak
        min_dist_hm = 4.0  # heatmap pixels (~16 image pixels)
        kept_peaks = []
        for row, col, conf in raw_peaks:
            is_dup = False
            for kr, kc, _ in kept_peaks:
                if (row - kr) ** 2 + (col - kc) ** 2 < min_dist_hm ** 2:
                    is_dup = True
                    break
            if not is_dup:
                kept_peaks.append((row, col, conf))

        detections = []
        for row, col, conf in kept_peaks[:self.max_detections]:
            # Sub-pixel refinement via weighted centroid in local 3×3 patch
            cx, cy = self._subpixel_refine(heatmap, row, col)

            # Map back to full image coordinates
            det_x = cx * scale_x
            det_y = cy * scale_y

            # Estimate bounding box from heatmap spread
            half_size = 8
            detections.append(Detection(
                x=det_x,
                y=det_y,
                confidence=conf,
                bbox_x=det_x - half_size,
                bbox_y=det_y - half_size,
                bbox_w=half_size * 2,
                bbox_h=half_size * 2,
                peak_intensity=int(conf * 255),
            ))

        return detections

    @staticmethod
    def _subpixel_refine(heatmap: "torch.Tensor", row: int, col: int) -> tuple:
        """Weighted centroid in a 3×3 window for sub-pixel peak localisation."""
        h, w = heatmap.shape
        total_w = 0.0
        sum_x = 0.0
        sum_y = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = row + dy, col + dx
                if 0 <= ny < h and 0 <= nx < w:
                    val = heatmap[ny, nx].item()
                    weight = val ** 2  # quadratic weighting for sharper localisation
                    total_w += weight
                    sum_x += nx * weight
                    sum_y += ny * weight
        if total_w > 0:
            return sum_x / total_w, sum_y / total_w
        return float(col), float(row)

    # ------------------------------------------------------------------
    # Classical Detection Path (fallback)
    # ------------------------------------------------------------------
    def _detect_classical(self, image: np.ndarray) -> List[Detection]:
        """Adaptive thresholding + connected components + spatial moments.

        Runs without PyTorch.  Robust for the expected domain (bright
        spots on near-black background) and serves as the performance
        baseline the CNN must beat.
        """
        h, w = image.shape[:2]
        detections: List[Detection] = []

        # 1. Compute adaptive threshold
        #    Use a noise-floor estimate from the image's median
        median_val = int(np.median(image))
        threshold = max(self._classical_threshold, median_val + 25)

        # 2. Binary mask
        binary = (image > threshold).astype(np.uint8)

        # 3. Morphological opening + connected components
        #    Use scipy when available (10–50x faster than pure Python)
        try:
            from scipy import ndimage
            struct = np.ones((3, 3), dtype=np.uint8)
            opened = ndimage.binary_opening(binary, structure=struct).astype(np.uint8)
            labels, num_labels = ndimage.label(opened, structure=struct)
        except ImportError:
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = self._morph_erode(binary, kernel)
            opened = self._morph_dilate(eroded, kernel)
            labels, num_labels = self._connected_components(opened, h, w)

        # 5. Extract centroids per component
        for label_id in range(1, num_labels + 1):
            component_mask = (labels == label_id)
            area = int(np.sum(component_mask))

            if area < self._min_blob_area or area > self._max_blob_area:
                continue

            # Weighted moment centroid (sub-pixel accuracy)
            ys, xs = np.nonzero(component_mask)
            weights = image[ys, xs].astype(np.float64)
            weights_pow = weights ** 1.4  # non-linear for sub-pixel precision

            total_w = np.sum(weights_pow)
            if total_w < 1e-8:
                continue

            cx = float(np.sum(xs * weights_pow) / total_w)
            cy = float(np.sum(ys * weights_pow) / total_w)

            # Bounding box
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            bbox_w = max(8, x_max - x_min + 4)
            bbox_h = max(8, y_max - y_min + 4)

            # Peak intensity
            peak = int(np.max(image[ys, xs]))

            # Confidence heuristic: SNR + compactness
            snr = peak / 255.0
            compactness = min(1.0, 30.0 / max(1, area))
            confidence = float(np.clip(snr * 0.65 + compactness * 0.35, 0.05, 0.99))

            detections.append(Detection(
                x=cx,
                y=cy,
                confidence=confidence,
                bbox_x=float(x_min - 2),
                bbox_y=float(y_min - 2),
                bbox_w=float(bbox_w),
                bbox_h=float(bbox_h),
                peak_intensity=peak,
            ))

        return detections

    # ------------------------------------------------------------------
    # Pure-numpy morphological ops (avoids OpenCV dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _morph_erode(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(binary, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
        h, w = binary.shape
        result = np.ones_like(binary)
        for dy in range(kh):
            for dx in range(kw):
                if kernel[dy, dx]:
                    result &= padded[dy:dy + h, dx:dx + w]
        return result

    @staticmethod
    def _morph_dilate(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(binary, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
        h, w = binary.shape
        result = np.zeros_like(binary)
        for dy in range(kh):
            for dx in range(kw):
                if kernel[dy, dx]:
                    result |= padded[dy:dy + h, dx:dx + w]
        return result

    @staticmethod
    def _connected_components(binary: np.ndarray, h: int, w: int):
        """Two-pass connected-component labelling (4-connectivity)."""
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        equivalences: dict = {}

        def find_root(lbl):
            while equivalences.get(lbl, lbl) != lbl:
                lbl = equivalences[lbl]
            return lbl

        # First pass
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 0:
                    continue
                neighbours = []
                if y > 0 and labels[y - 1, x] > 0:
                    neighbours.append(labels[y - 1, x])
                if x > 0 and labels[y, x - 1] > 0:
                    neighbours.append(labels[y, x - 1])

                if not neighbours:
                    current_label += 1
                    labels[y, x] = current_label
                    equivalences[current_label] = current_label
                else:
                    min_label = min(find_root(n) for n in neighbours)
                    labels[y, x] = min_label
                    for n in neighbours:
                        root = find_root(n)
                        if root != min_label:
                            equivalences[root] = min_label

        # Second pass — resolve equivalences
        for y in range(h):
            for x in range(w):
                if labels[y, x] > 0:
                    labels[y, x] = find_root(labels[y, x])

        # Relabel to contiguous IDs
        unique = np.unique(labels[labels > 0])
        remap = {old: i + 1 for i, old in enumerate(unique)}
        for y in range(h):
            for x in range(w):
                if labels[y, x] > 0:
                    labels[y, x] = remap[labels[y, x]]

        return labels, len(remap)
