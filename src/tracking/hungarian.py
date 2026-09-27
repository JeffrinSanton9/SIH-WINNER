"""
SADHA — Module 2: Hungarian Bipartite Data Association & Gating Engine
Optimal global one-to-one assignment between predicted Kalman tracks and NN1 detections.

[INTENTIONAL DESIGN DECISION]: Cost Matrix Construction & Distance Gating Threshold
In multi-beacon coarse acquisition, ambiguous pairings occur when beacons cross paths
or when false detections arise from flare/clutter. 
1. Cost Matrix: Constructed using 2D Euclidean spatial distances between each track's
   Kalman-predicted position and newly observed NN1 candidate centroids:
     C_{i, j} = || p_pred_i - z_det_j ||_2
2. Gated Rejection: While the classical Hungarian algorithm finds the global minimum sum
   of costs across the bipartite graph, it would unconditionally pair a track with a far-away
   detection if no close candidate exists. Distance gating sets a hard upper bound:
   any pairing where C_{i, j} > gate_threshold is rejected.
3. Threshold Rationale: A generous threshold (e.g. 160 px in MultiBeaconTracker) is chosen
   to accommodate extreme accelerations and platform vibrations at 30 Hz without dropping
   true tracks, while rejecting absurd cross-canvas matches.

Backend:
Dispatches to SciPy's O(N^3) modified Jonker-Volgenant / LAPMOD implementation when available,
with a robust fallback to a pure-Python Munkres implementation if SciPy is absent.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def solve_assignment(
    cost_matrix: np.ndarray,
    gate_threshold: float = 80.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve the linear sum assignment problem with post-optimization distance gating.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (num_tracks, num_detections), where entry [i, j] represents
        the Euclidean distance in pixels between track i's prediction and detection j.
    gate_threshold : float
        Maximum spatial gating distance in pixels. Pairings with cost > gate_threshold
        are rejected, treating the track as unmatched (triggering miss counter) and the
        detection as unmatched (potentially spawning a new track).

    Returns
    -------
    matches : List[Tuple[int, int]]
        List of accepted (track_index, detection_index) coordinate tuples.
    unmatched_tracks : List[int]
        Indices of tracks that had no detection within gate_threshold.
    unmatched_detections : List[int]
        Indices of detections that were not assigned to any existing track.
    """
    num_tracks, num_dets = cost_matrix.shape

    if num_tracks == 0:
        return [], [], list(range(num_dets))
    if num_dets == 0:
        return [], list(range(num_tracks)), []

    # Solve optimal assignment
    row_indices, col_indices = _linear_sum_assignment(cost_matrix)

    # Gate: reject assignments exceeding threshold
    matches = []
    unmatched_tracks = set(range(num_tracks))
    unmatched_dets = set(range(num_dets))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= gate_threshold:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def build_cost_matrix(
    track_positions: List[Tuple[float, float]],
    detection_positions: List[Tuple[float, float]],
) -> np.ndarray:
    """Build Euclidean distance cost matrix.

    Parameters
    ----------
    track_positions : list of (x, y)
        Predicted positions of existing tracks.
    detection_positions : list of (x, y)
        NN1 detection centroids this frame.

    Returns
    -------
    np.ndarray of shape (num_tracks, num_detections)
    """
    nt = len(track_positions)
    nd = len(detection_positions)
    cost = np.zeros((nt, nd), dtype=np.float64)
    for i, (tx, ty) in enumerate(track_positions):
        for j, (dx, dy) in enumerate(detection_positions):
            cost[i, j] = math.sqrt((tx - dx) ** 2 + (ty - dy) ** 2)
    return cost


# ------------------------------------------------------------------
# Backend: scipy or pure-Python fallback
# ------------------------------------------------------------------
def _linear_sum_assignment(cost: np.ndarray):
    """Dispatch to scipy or fallback Munkres."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        return _munkres_fallback(cost)


def _munkres_fallback(cost: np.ndarray):
    """Pure-Python Hungarian / Munkres algorithm for small matrices.

    Handles rectangular matrices by padding to square.
    """
    n = max(cost.shape)
    padded = np.full((n, n), cost.max() * 10, dtype=np.float64)
    padded[:cost.shape[0], :cost.shape[1]] = cost

    # Step 1: Row reduction
    c = padded.copy()
    for i in range(n):
        c[i] -= c[i].min()
    # Step 2: Column reduction
    for j in range(n):
        c[:, j] -= c[:, j].min()

    # Iterative covering and augmentation
    max_iter = n * n * 2
    for _ in range(max_iter):
        # Find assignment using zeros
        assignment = _find_assignment(c, n)
        if assignment is not None:
            # Filter to original dimensions
            rows = [r for r, c_ in assignment if r < cost.shape[0] and c_ < cost.shape[1]]
            cols = [c_ for r, c_ in assignment if r < cost.shape[0] and c_ < cost.shape[1]]
            return np.array(rows), np.array(cols)

        # Cover zeros with minimum lines and adjust
        row_cover, col_cover = _min_cover(c, n)
        # Find minimum uncovered value
        min_val = np.inf
        for i in range(n):
            for j in range(n):
                if not row_cover[i] and not col_cover[j]:
                    min_val = min(min_val, c[i, j])
        if min_val == np.inf or min_val == 0:
            break
        # Subtract from uncovered, add to double-covered
        for i in range(n):
            for j in range(n):
                if not row_cover[i] and not col_cover[j]:
                    c[i, j] -= min_val
                elif row_cover[i] and col_cover[j]:
                    c[i, j] += min_val

    # Fallback: greedy assignment
    return _greedy_assignment(cost)


def _find_assignment(c: np.ndarray, n: int):
    """Try to find a complete assignment using only zero entries."""
    assignment = []
    row_used = [False] * n
    col_used = [False] * n

    def _recurse(row):
        if row == n:
            return True
        if row_used[row]:
            return _recurse(row + 1)
        for col in range(n):
            if not col_used[col] and abs(c[row, col]) < 1e-10:
                col_used[col] = True
                row_used[row] = True
                assignment.append((row, col))
                if _recurse(row + 1):
                    return True
                assignment.pop()
                col_used[col] = False
                row_used[row] = False
        return False

    if _recurse(0):
        return assignment
    return None


def _min_cover(c: np.ndarray, n: int):
    """Find minimum lines to cover all zeros (heuristic)."""
    row_cover = [False] * n
    col_cover = [False] * n

    # Greedy: cover rows/cols with most zeros
    for _ in range(n):
        best_type = None
        best_idx = -1
        best_count = -1

        for i in range(n):
            if row_cover[i]:
                continue
            count = sum(1 for j in range(n) if not col_cover[j] and abs(c[i, j]) < 1e-10)
            if count > best_count:
                best_count = count
                best_idx = i
                best_type = 'row'

        for j in range(n):
            if col_cover[j]:
                continue
            count = sum(1 for i in range(n) if not row_cover[i] and abs(c[i, j]) < 1e-10)
            if count > best_count:
                best_count = count
                best_idx = j
                best_type = 'col'

        if best_count <= 0:
            break
        if best_type == 'row':
            row_cover[best_idx] = True
        else:
            col_cover[best_idx] = True

    return row_cover, col_cover


def _greedy_assignment(cost: np.ndarray):
    """Greedy nearest-first assignment (last-resort fallback)."""
    nr, nc = cost.shape
    used_cols = set()
    rows, cols = [], []

    flat = []
    for i in range(nr):
        for j in range(nc):
            flat.append((cost[i, j], i, j))
    flat.sort()

    used_rows = set()
    for _, r, c in flat:
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)

    return np.array(rows), np.array(cols)
