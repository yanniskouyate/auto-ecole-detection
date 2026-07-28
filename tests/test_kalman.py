"""Tests unitaires — filtre de Kalman et multi-tracker."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.tracking.kalman_filter import KalmanCV2D  # noqa: E402
from auto_ecole_math.tracking.multi_tracker import Detection, MultiObjectTracker  # noqa: E402


def test_kalman_tracks_constant_velocity():
    dt = 0.1
    kf = KalmanCV2D(dt=dt, process_var=0.1, meas_var=0.01)
    # True motion: X = 0.5 t, Y = 2 + 1.0 t
    truth = []
    estimates = []
    for k in range(40):
        t = k * dt
        true_xy = np.array([0.5 * t, 2.0 + 1.0 * t])
        meas = true_xy + np.random.default_rng(k).normal(0, 0.05, size=2)
        if k == 0:
            kf.initiate(meas)
        else:
            kf.predict(dt)
            kf.update(meas)
        truth.append(true_xy)
        estimates.append(kf.position.copy())
    err = np.linalg.norm(np.array(estimates[-10:]) - np.array(truth[-10:]), axis=1).mean()
    assert err < 0.15
    # Velocity should approach (0.5, 1.0)
    assert kf.velocity[0] == pytest.approx(0.5, abs=0.25)
    assert kf.velocity[1] == pytest.approx(1.0, abs=0.25)


def test_kalman_mahalanobis_near_zero_for_consistent_meas():
    kf = KalmanCV2D(dt=0.1, meas_var=0.25)
    kf.initiate(np.array([1.0, 2.0]))
    kf.predict()
    d = kf.mahalanobis(np.array([1.0, 2.0]))
    assert d < 1.0


def test_multi_tracker_assigns_stable_id():
    tracker = MultiObjectTracker(max_age=5, min_hits=2, gate=3.0, use_mahalanobis=False, dt=0.1)
    confirmed_ids = []
    for k in range(10):
        xy = np.array([1.0 + 0.1 * k, 5.0 + 0.2 * k])
        dets = [Detection(xy=xy, label="person", confidence=0.9)]
        tracks = tracker.update(dets, dt=0.1)
        if tracks:
            confirmed_ids.append(tracks[0].track_id)
    assert len(set(confirmed_ids)) == 1


def test_multi_tracker_coasts_through_occlusion():
    tracker = MultiObjectTracker(max_age=5, min_hits=2, gate=5.0, use_mahalanobis=False, dt=0.1)
    # Warm-up
    for k in range(5):
        tracker.update(
            [Detection(xy=np.array([0.0, float(k)]), label="car", confidence=0.8)],
            dt=0.1,
        )
    # Occlusion for 3 frames
    for _ in range(3):
        tracks = tracker.update([], dt=0.1)
    assert len(tracker.tracks) >= 1
    assert tracker.tracks[0].time_since_update == 3
