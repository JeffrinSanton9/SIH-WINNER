"""
SADHA — NN1 Standalone Test & Training Script

Run this to:
  1. Validate the classical detector on synthetic test cases.
  2. Train the CNN model on synthetic data.
  3. Compare CNN vs classical accuracy.

Usage:
    python -m src.tracking.nn1_test             # Run tests only
    python -m src.tracking.nn1_test --train      # Train + test
    python -m src.tracking.nn1_test --train --epochs 50
"""

from __future__ import annotations

import argparse
import time
import math
import numpy as np

from src.core.frame_data import FrameData
from src.tracking.nn1_types import NN1Result
from src.tracking.nn1_detector import NN1Detector


def make_test_frame(
    beacons: list,
    noise_std: float = 0.0,
    salt_pepper: float = 0.0,
    w: int = 640,
    h: int = 480,
) -> tuple:
    """Create a synthetic test frame with beacons at given positions.

    Returns (FrameData, ground_truth_positions).
    """
    rng = np.random.default_rng(42)
    frame = np.full((h, w), 6, dtype=np.uint8)

    for bx, by, size in beacons:
        half = size // 2
        y0 = max(0, int(by) - half)
        y1 = min(h, int(by) + half + 1)
        x0 = max(0, int(bx) - half)
        x1 = min(w, int(bx) + half + 1)
        frame[y0:y1, x0:x1] = 230

    if noise_std > 0:
        noise = rng.normal(0, noise_std, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if salt_pepper > 0:
        mask = rng.random(frame.shape)
        frame[mask < salt_pepper / 2] = 0
        frame[mask > 1 - salt_pepper / 2] = 255

    fd = FrameData(frame_index=1, timestamp=0.033, image=frame)
    gt = [(bx, by) for bx, by, _ in beacons]
    return fd, gt


def evaluate_detector(detector: NN1Detector, label: str):
    """Run a battery of test cases and report accuracy."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    tests = [
        ("Single beacon, clean",
         [(300, 200, 10)], 0.0, 0.0),
        ("Single beacon, Gaussian noise std=15",
         [(300, 200, 10)], 15.0, 0.0),
        ("Single beacon, salt & pepper 8%",
         [(300, 200, 10)], 0.0, 0.08),
        ("Single beacon, combined noise",
         [(300, 200, 10)], 12.0, 0.05),
        ("Small beacon (5px), clean",
         [(400, 250, 5)], 0.0, 0.0),
        ("Large beacon (18px), noise",
         [(200, 350, 18)], 10.0, 0.03),
        ("Two beacons, clean",
         [(150, 100, 10), (480, 360, 10)], 0.0, 0.0),
        ("Three beacons, noise",
         [(100, 80, 8), (320, 240, 12), (550, 400, 10)], 10.0, 0.04),
        ("Empty frame (no beacon)",
         [], 5.0, 0.02),
    ]

    total_tests = len(tests)
    passed = 0
    total_error = 0.0
    total_detections = 0

    for name, beacons, noise, sp in tests:
        fd, gt = make_test_frame(beacons, noise, sp)
        result = detector.detect(fd)

        # Check detection count
        expected_count = len(beacons)
        actual_count = result.num_detections
        count_ok = actual_count == expected_count

        # Check positional accuracy (match each detection to nearest GT)
        max_err = 0.0
        if expected_count > 0 and actual_count > 0:
            for gx, gy in gt:
                best = min(
                    math.sqrt((d.x - gx)**2 + (d.y - gy)**2)
                    for d in result.detections
                )
                max_err = max(max_err, best)
                total_error += best
                total_detections += 1

        err_ok = max_err <= 10.0 or expected_count == 0
        ok = count_ok and err_ok

        status = "PASS" if ok else "FAIL"
        if passed or ok:
            passed += 1 if ok else 0

        det_str = ", ".join(
            f"({d.x:.1f},{d.y:.1f} c={d.confidence:.2f})"
            for d in result.detections
        ) or "(none)"

        print(f"  [{status}] {name}")
        print(f"        Expected {expected_count} det, got {actual_count} — "
              f"max err {max_err:.1f}px")
        print(f"        Detections: {det_str}")

    avg_err = total_error / max(1, total_detections)
    print(f"\n  Results: {passed}/{total_tests} passed, "
          f"avg positional error = {avg_err:.2f} px")

    # Speed benchmark
    fd_bench, _ = make_test_frame([(300, 200, 10)], 10.0, 0.03)
    t0 = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        detector.detect(fd_bench)
    elapsed = time.perf_counter() - t0
    fps = n_iters / elapsed
    print(f"  Speed: {fps:.1f} FPS ({elapsed/n_iters*1000:.1f} ms/frame)")

    return passed, total_tests, avg_err


def main():
    parser = argparse.ArgumentParser(description="NN1 Test & Training")
    parser.add_argument("--train", action="store_true", help="Train CNN model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    # Test classical detector
    det_classical = NN1Detector(mode="classical")
    evaluate_detector(det_classical, "CLASSICAL DETECTOR (baseline)")

    # Train CNN if requested
    if args.train:
        try:
            from src.tracking.nn1_trainer import NN1Trainer
            trainer = NN1Trainer()
            trainer.train(epochs=args.epochs, samples_per_epoch=args.samples)

            # Test CNN detector with fresh weights
            det_cnn = NN1Detector(mode="cnn")
            evaluate_detector(det_cnn, "CNN DETECTOR (post-training)")
        except ImportError:
            print("\nPyTorch not available — skipping CNN training.")
    else:
        # Try CNN if weights exist
        try:
            det_auto = NN1Detector(mode="auto")
            if det_auto._use_cnn:
                evaluate_detector(det_auto, "CNN DETECTOR (pre-trained weights)")
            else:
                print("\nNo CNN weights found — run with --train to train the model.")
        except Exception:
            print("\nCNN not available — classical detector is the active backend.")


if __name__ == "__main__":
    main()
