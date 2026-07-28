"""Tests unitaires — homographie plan-sol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.geometry.calibration import (  # noqa: E402
    HomographyCalibration,
    load_calibration,
    save_calibration,
)
from auto_ecole_math.geometry.homography import (  # noqa: E402
    HomographyEstimator,
    bbox_bottom_center,
)


def test_bbox_bottom_center():
    uv = bbox_bottom_center([10, 20, 30, 40])
    assert uv[0] == pytest.approx(20.0)
    assert uv[1] == pytest.approx(40.0)


def test_load_default_calibration():
    path = ROOT / "config" / "homography_default.json"
    calib = load_calibration(path)
    assert calib.image_points.shape[0] >= 4
    assert calib.ground_points.shape == calib.image_points.shape


def test_homography_roundtrip_on_calibration_points():
    est = HomographyEstimator.from_json(ROOT / "config" / "homography_default.json")
    img = est.calibration.image_points
    grd = est.pixels_to_ground(img)
    # Ground points should match calibration within RANSAC tolerance (metres)
    err = np.linalg.norm(grd - est.calibration.ground_points, axis=1)
    assert float(np.max(err)) < 0.05
    # Round-trip ground → pixel → ground
    back_px = est.grounds_to_pixels(grd)
    back_grd = est.pixels_to_ground(back_px)
    assert float(np.max(np.linalg.norm(back_grd - grd, axis=1))) < 1e-3


def test_known_ground_distance():
    est = HomographyEstimator.from_json(ROOT / "config" / "homography_default.json")
    # Calibration rectangle: width 3.5 m between (±1.75, 4)
    p1 = np.array([-1.75, 4.0])
    p2 = np.array([1.75, 4.0])
    d = HomographyEstimator.ground_distance(p1, p2)
    assert d == pytest.approx(3.5, abs=1e-9)


def test_save_load_roundtrip(tmp_path):
    calib = load_calibration(ROOT / "config" / "homography_default.json")
    out = tmp_path / "calib.json"
    save_calibration(calib, out)
    calib2 = load_calibration(out)
    np.testing.assert_allclose(calib.image_points, calib2.image_points)
    np.testing.assert_allclose(calib.ground_points, calib2.ground_points)


def test_mean_reprojection_error_small():
    est = HomographyEstimator.from_json(ROOT / "config" / "homography_default.json")
    assert est.mean_reprojection_error() < 0.05


def test_calibration_requires_four_points():
    with pytest.raises(ValueError):
        HomographyCalibration(
            image_points=np.zeros((3, 2)),
            ground_points=np.zeros((3, 2)),
        )
