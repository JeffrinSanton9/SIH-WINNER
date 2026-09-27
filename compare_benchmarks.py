"""
SADHA vs. Conventional ATP Baseline — Head-to-Head Comparative Benchmark Suite

Evaluates both architectures across all 5 standardized SIH Benchmark-1 scenarios:
  1. NOMINAL_LOW_DYNAMICS   - Baseline steady tracking
  2. HIGH_DYNAMICS_FIG8     - High acceleration & camera jitter
  3. SEVERE_FOG_NOISE       - Heavy fog, 8% salt & pepper noise, low-light
  4. MULTI_BEACON_CROSSING  - Proximity crossing & data association
  5. SPIRAL_AGGRESSIVE      - Expanding spiral, rain, multi-axis platform motion

Computes evaluator-grade performance metrics:
  - Mean Tracking Error (RMSE in pixels)
  - Max Tracking Error (pixels)
  - Target Retention / Lock Rate (%)
  - Target Loss Rate (%)
  - Initial Acquisition Time (seconds)
  - Mean Re-acquisition Time (seconds)
  - Identity Switches / ID Swaps
  - Processing Throughput (FPS)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.config import SimulationConfig, apply_preset, PRESET_SCENARIOS
from src.core.frame_data import FrameData, BeaconGroundTruth
from src.sim.world import WorldSimulation

# SADHA Architecture
from src.tracking.nn1_detector import NN1Detector
from src.tracking.tracker import MultiBeaconTracker
from src.tracking.nn2_predictor import NN2Predictor
from src.tracking.tracker_types import LockState

# Conventional Baseline Architecture
from baseline.detector import BaselineDetector
from baseline.tracker import BaselineTracker


@dataclass
class ScenarioMetrics:
    system_name: str
    scenario_name: str
    total_frames: int
    mean_tracking_error_px: float
    max_tracking_error_px: float
    lock_rate_pct: float
    loss_rate_pct: float
    acquisition_time_sec: float
    mean_reacquisition_time_sec: float
    id_switches: int
    mean_fps: float
    precision_pct: float
    recall_pct: float


def run_system_evaluation(
    system_type: str,  # "SADHA" or "BASELINE"
    preset_key: str,
    duration_sec: float = 12.0,
    sim_dt: float = 1.0 / 60.0,
) -> ScenarioMetrics:
    """Run a closed-loop simulation evaluation for the given system and scenario."""
    # Deterministic seed for reproducible evaluation
    np.random.seed(42)

    # 1. Initialize World Simulation with Preset
    cfg = apply_preset(preset_key)
    world = WorldSimulation(cfg)

    # 2. Initialize Pipeline
    if system_type == "SADHA":
        detector = NN1Detector(mode="auto")
        tracker = MultiBeaconTracker()
        predictor = NN2Predictor(mode="auto")
    else:
        detector = BaselineDetector()
        tracker = BaselineTracker()
        predictor = None

    # Tracking metrics accumulators
    tracking_errors: List[float] = []
    locked_frames = 0
    lost_frames = 0
    total_sensor_frames = 0

    acquisition_time: Optional[float] = None
    loss_start_time: Optional[float] = None
    reacquisition_durations: List[float] = []

    # ID switch tracking: map track_id -> ground_truth beacon_id
    track_to_gt_map: Dict[int, int] = {}
    id_switches = 0

    # Detection quality metrics (IoU / proximity matching)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    processing_times: List[float] = []
    sim_time = 0.0

    while sim_time < duration_sec:
        # Step simulation physics
        frame_data: Optional[FrameData] = world.step(sim_dt)
        sim_time += sim_dt

        if frame_data is None:
            continue

        total_sensor_frames += 1
        t_start = time.perf_counter()

        # ---- Perception Stage ----
        nn1_res = detector.detect(frame_data)

        # ---- Association & Tracking Stage ----
        tracker_res = tracker.update(nn1_res)

        # ---- Gimbal Control & Re-acquisition Actuation ----
        cmds = tracker.compute_pan_tilt_commands()
        if cmds:
            chosen_term = 0
            for trk in tracker_res.tracks:
                if trk.lock_state == LockState.LOCKED:
                    chosen_term = trk.terminal_id
                    break
            if chosen_term in cmds:
                pan_cmd, tilt_cmd = cmds[chosen_term]
            else:
                pan_cmd, tilt_cmd = next(iter(cmds.values()))
            world.camera.set_commanded_rates(pan_cmd, tilt_cmd)

        # Dynamic Loss-Ramping (SADHA specific feature)
        is_any_lost = any(trk.lock_state == LockState.LOST for trk in tracker_res.tracks)
        if system_type == "SADHA":
            if is_any_lost:
                world.camera.set_capture_rate(60.0)
            else:
                world.camera.set_capture_rate(30.0)

            # Optional NN2 prediction cadence
            if predictor is not None:
                predictor.update(tracker_res, frame_data.frame_index, frame_data.timestamp)

        t_end = time.perf_counter()
        processing_times.append(t_end - t_start)

        # ---- Ground Truth Evaluation ----
        gt_beacons = {b.beacon_id: b for b in frame_data.ground_truth if b.in_fov}
        active_tracks = {trk.track_id: trk for trk in tracker_res.tracks}

        # Check lock states
        has_locked = any(trk.lock_state == LockState.LOCKED for trk in tracker_res.tracks)
        has_lost = any(trk.lock_state == LockState.LOST for trk in tracker_res.tracks)

        if has_locked:
            locked_frames += 1
            # Check initial acquisition
            if acquisition_time is None:
                acquisition_time = sim_time
            # Check re-acquisition
            if loss_start_time is not None:
                reacq_dur = sim_time - loss_start_time
                reacquisition_durations.append(reacq_dur)
                loss_start_time = None
        else:
            if has_lost:
                lost_frames += 1
                if loss_start_time is None and acquisition_time is not None:
                    loss_start_time = sim_time

        # Match tracks to ground truth to compute tracking error and ID switches
        for tid, trk in active_tracks.items():
            tx, ty = trk.position
            # Find nearest ground truth beacon
            best_gt_id = -1
            best_dist = float("inf")
            for bid, gt in gt_beacons.items():
                d = math.sqrt((tx - gt.u) ** 2 + (ty - gt.v) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_gt_id = bid

            if best_dist < 40.0 and best_gt_id >= 0:
                tracking_errors.append(best_dist)
                # Check for ID swap
                if tid in track_to_gt_map:
                    if track_to_gt_map[tid] != best_gt_id:
                        id_switches += 1
                        track_to_gt_map[tid] = best_gt_id
                else:
                    track_to_gt_map[tid] = best_gt_id

        # Detection Precision / Recall
        detected_coords = [(d.x, d.y) for d in nn1_res.detections]
        gt_coords = [(b.u, b.v) for b in gt_beacons.values()]

        matched_gt = set()
        for dx, dy in detected_coords:
            match = False
            for g_idx, (gu, gv) in enumerate(gt_coords):
                if g_idx not in matched_gt and math.sqrt((dx - gu)**2 + (dy - gv)**2) <= 15.0:
                    matched_gt.add(g_idx)
                    match = True
                    break
            if match:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += len(gt_coords) - len(matched_gt)

    # Compute Aggregate Metrics
    mean_err = float(np.mean(tracking_errors)) if tracking_errors else 99.9
    max_err = float(np.max(tracking_errors)) if tracking_errors else 99.9
    lock_pct = (locked_frames / max(1, total_sensor_frames)) * 100.0
    loss_pct = (lost_frames / max(1, total_sensor_frames)) * 100.0

    acq_time = acquisition_time if acquisition_time is not None else duration_sec
    mean_reacq = float(np.mean(reacquisition_durations)) if reacquisition_durations else 0.0

    mean_latency = float(np.mean(processing_times)) if processing_times else 0.033
    mean_fps = 1.0 / max(1e-4, mean_latency)

    precision = (true_positives / max(1, true_positives + false_positives)) * 100.0
    recall = (true_positives / max(1, true_positives + false_negatives)) * 100.0

    return ScenarioMetrics(
        system_name=system_type,
        scenario_name=PRESET_SCENARIOS[preset_key]["name"],
        total_frames=total_sensor_frames,
        mean_tracking_error_px=round(mean_err, 2),
        max_tracking_error_px=round(max_err, 2),
        lock_rate_pct=round(lock_pct, 1),
        loss_rate_pct=round(loss_pct, 1),
        acquisition_time_sec=round(acq_time, 3),
        mean_reacquisition_time_sec=round(mean_reacq, 3),
        id_switches=id_switches,
        mean_fps=round(mean_fps, 1),
        precision_pct=round(precision, 1),
        recall_pct=round(recall, 1),
    )


def run_all_benchmarks():
    """Execute head-to-head comparison across all benchmark scenarios."""
    scenarios = [
        "NOMINAL_LOW_DYNAMICS",
        "HIGH_DYNAMICS_FIG8",
        "SEVERE_FOG_NOISE",
        "MULTI_BEACON_CROSSING",
        "SPIRAL_AGGRESSIVE",
    ]

    print("=" * 80)
    print(" SADHA vs. CONVENTIONAL BASELINE -- FULL BENCHMARK EVALUATION")
    print("=" * 80)

    results_sadha: List[ScenarioMetrics] = []
    results_baseline: List[ScenarioMetrics] = []

    for sc_key in scenarios:
        sc_title = PRESET_SCENARIOS[sc_key]["name"]
        print(f"\n[RUNNING BENCHMARK] {sc_title} ...", flush=True)

        # 1. Baseline
        t0 = time.time()
        base_res = run_system_evaluation("BASELINE", sc_key, duration_sec=6.0)
        results_baseline.append(base_res)
        print(f"  [+] Baseline completed in {time.time() - t0:.1f}s | Err: {base_res.mean_tracking_error_px} px | Loss: {base_res.loss_rate_pct}%", flush=True)

        # 2. SADHA
        t0 = time.time()
        sadha_res = run_system_evaluation("SADHA", sc_key, duration_sec=6.0)
        results_sadha.append(sadha_res)
        print(f"  [+] SADHA completed in {time.time() - t0:.1f}s    | Err: {sadha_res.mean_tracking_error_px} px | Loss: {sadha_res.loss_rate_pct}%", flush=True)

    # Output formatted report
    print("\n" + "=" * 90, flush=True)
    print(f"{'SCENARIO':<32} | {'SYSTEM':<9} | {'ERR (px)':<9} | {'LOCK %':<7} | {'LOSS %':<7} | {'ID SWAPS':<8} | {'FPS':<6}", flush=True)
    print("=" * 90, flush=True)

    for base, sadha in zip(results_baseline, results_sadha):
        name = base.scenario_name.replace("Benchmark-1 · ", "").replace("Benchmark-1 - ", "")
        print(f"{name:<32} | {'Baseline':<9} | {base.mean_tracking_error_px:<9} | {base.lock_rate_pct:<7} | {base.loss_rate_pct:<7} | {base.id_switches:<8} | {base.mean_fps:<6}", flush=True)
        print(f"{'':<32} | {'SADHA':<9} | {sadha.mean_tracking_error_px:<9} | {sadha.lock_rate_pct:<7} | {sadha.loss_rate_pct:<7} | {sadha.id_switches:<8} | {sadha.mean_fps:<6}", flush=True)
        print("-" * 90, flush=True)

    # Save to disk as benchmark_results.json
    all_data = {
        "baseline": [asdict(r) for r in results_baseline],
        "sadha": [asdict(r) for r in results_sadha],
    }
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print("\n[SUCCESS] Full benchmark results exported to benchmark_results.json", flush=True)


if __name__ == "__main__":
    run_all_benchmarks()
