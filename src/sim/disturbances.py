"""
SADHA — Module 1: Disturbances & Noise Simulation Engine
Applies intensity-domain degradation and spatial focal plane jitter to sensor frames.

[INTENTIONAL DESIGN DECISION]: Intensity-Domain Noise vs. Spatial Camera Jitter
In optical sensor modeling, thermal detector noise (Gaussian) and photon arrival
fluctuations (Poisson) corrupt electron well counts per pixel — an intensity-domain
phenomenon that modulates pixel Digital Numbers (DN) between 0 and 255 without
displacing the optical image geometry. Conversely, structural mount vibrations,
acoustic buffeting, and servo hunting physically displace the optical focal plane,
producing spatial coordinate shifts (dx, dy).
To maintain rigorous physical fidelity:
  1. Gaussian and Poisson noise are applied strictly to pixel intensities. The SIH
     spec's "20 px std dev" figure is properly interpreted as an intensity-domain
     noise amplitude cap (sigma up to 20 DN on a 0-255 scale) to avoid distorting image
     dimensions.
  2. Physical mount vibration is modeled independently by the spatial Camera Jitter
     engine as real pixel-coordinate translations (jitter_x, jitter_y) on the sensor plane.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

from src.core.types import AtmosphericCondition


class DisturbanceEngine:
    """Applies all optical, atmospheric, and sensor noise effects to monochrome frames."""

    def __init__(
        self,
        salt_pepper_enabled: bool = False,
        salt_pepper_density: float = 0.08,
        gaussian_enabled: bool = False,
        gaussian_std: float = 18.0,
        poisson_enabled: bool = False,
        poisson_scale: float = 1.0,
        jitter_enabled: bool = False,
        jitter_amplitude: float = 6.0,
        atmospheric: AtmosphericCondition = AtmosphericCondition.CLEAR,
    ):
        self.salt_pepper_enabled = salt_pepper_enabled
        self.salt_pepper_density = salt_pepper_density
        self.gaussian_enabled = gaussian_enabled
        self.gaussian_std = gaussian_std
        self.poisson_enabled = poisson_enabled
        self.poisson_scale = poisson_scale
        self.jitter_enabled = jitter_enabled
        self.jitter_amplitude = min(20.0, jitter_amplitude)
        self.atmospheric = atmospheric

        # Spatial camera jitter offset for current frame
        self.jitter_x = 0.0
        self.jitter_y = 0.0

    # ------------------------------------------------------------------
    # Per-frame update (spatial jitter)
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.jitter_enabled and self.jitter_amplitude > 0:
            self.jitter_x = random.uniform(-1, 1) * self.jitter_amplitude
            self.jitter_y = random.uniform(-1, 1) * self.jitter_amplitude
        else:
            self.jitter_x = 0.0
            self.jitter_y = 0.0

    # ------------------------------------------------------------------
    # Apply all image disturbances to a monochrome frame
    # ------------------------------------------------------------------
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Corrupt *frame* (H×W uint8 monochrome) in-place and return it.

        Application order:
          1. Atmospheric extinction / path radiance
          2. Gaussian intensity noise
          3. Poisson shot noise
          4. Salt-and-pepper pixel corruption
        """
        h, w = frame.shape[:2]
        img = frame.astype(np.float32)

        # ---- 1. Atmospheric degradation ----
        contrast, brightness, scatter = self._atmo_params()
        if self.atmospheric != AtmosphericCondition.CLEAR:
            img = (img - 128.0) * contrast + 128.0 + brightness
            if scatter > 0:
                img += np.random.uniform(-scatter, scatter, img.shape).astype(np.float32)

        # ---- 2. Gaussian intensity noise ----
        if self.gaussian_enabled and self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, img.shape).astype(np.float32)
            img += noise

        # ---- 3. Poisson shot noise ----
        if self.poisson_enabled:
            # Scale image to a photon-count regime, apply Poisson, scale back
            scale = 0.15
            positive = np.clip(img * scale, 0, None)
            noisy = np.random.poisson(positive.astype(np.float64)).astype(np.float32) / scale
            img = img * 0.6 + noisy * 0.4

        # ---- 4. Salt-and-pepper noise ----
        if self.salt_pepper_enabled and self.salt_pepper_density > 0:
            mask = np.random.random(img.shape)
            half = self.salt_pepper_density * 0.5
            img[mask < half] = 0.0              # pepper
            img[mask > 1.0 - half] = 255.0      # salt

        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Disturbance descriptor for NN2 input history
    # ------------------------------------------------------------------
    def get_descriptor(self) -> List[float]:
        """Returns a 5-element vector encoding the current noise state."""
        atmo_level = {
            AtmosphericCondition.CLEAR: 0.0,
            AtmosphericCondition.HAZE: 0.3,
            AtmosphericCondition.FOG: 0.8,
            AtmosphericCondition.RAIN: 0.6,
            AtmosphericCondition.LOW_LIGHT: 0.7,
        }.get(self.atmospheric, 0.0)

        return [
            self.salt_pepper_density if self.salt_pepper_enabled else 0.0,
            (self.gaussian_std / 50.0) if self.gaussian_enabled else 0.0,
            1.0 if self.poisson_enabled else 0.0,
            (self.jitter_amplitude / 20.0) if self.jitter_enabled else 0.0,
            atmo_level,
        ]

    # ------------------------------------------------------------------
    # Atmospheric parameter lookup
    # ------------------------------------------------------------------
    def _atmo_params(self):
        """Returns (contrast_factor, brightness_offset, scatter_amplitude)."""
        return {
            AtmosphericCondition.CLEAR:     (1.0,  0.0,  0.0),
            AtmosphericCondition.HAZE:      (0.65, 18.0, 0.0),
            AtmosphericCondition.FOG:       (0.35, 45.0, 9.0),
            AtmosphericCondition.RAIN:      (0.55, 15.0, 0.0),
            AtmosphericCondition.LOW_LIGHT: (0.40, -30.0, 0.0),
        }.get(self.atmospheric, (1.0, 0.0, 0.0))
