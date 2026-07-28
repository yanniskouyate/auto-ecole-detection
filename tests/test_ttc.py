"""Tests unitaires — TTC et freinage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.kinematics.braking import (  # noqa: E402
    braking_distance,
    stopping_margin,
    two_second_rule_distance,
)
from auto_ecole_math.kinematics.ttc import (  # noqa: E402
    relative_closing_rate,
    time_to_collision,
    ttc_from_track,
)


def test_ttc_closing():
    # d=20 m, closing at 10 m/s → TTC = 2 s
    assert time_to_collision(20.0, -10.0) == pytest.approx(2.0)


def test_ttc_diverging_is_inf():
    assert time_to_collision(20.0, 1.0) == float("inf")


def test_relative_closing_rate():
    d, dd = relative_closing_rate(
        position=(0.0, 20.0),
        velocity=(0.0, -5.0),
        ego_position=(0.0, 0.0),
        ego_velocity=(0.0, 0.0),
    )
    assert d == pytest.approx(20.0)
    assert dd == pytest.approx(-5.0)
    assert time_to_collision(d, dd) == pytest.approx(4.0)


def test_ttc_from_track_range():
    info = ttc_from_track([0.0, 30.0], [0.0, -10.0])
    assert info["ttc_s"] == pytest.approx(3.0)


def test_braking_distance_formula():
    # v=10 m/s, tr=1 s, a=5 → 10 + 100/10 = 20
    assert braking_distance(10.0, reaction_time_s=1.0, deceleration_mps2=5.0) == pytest.approx(20.0)


def test_stopping_margin_unsafe():
    m = stopping_margin(10.0, 10.0, reaction_time_s=1.0, deceleration_mps2=5.0)
    assert m["d_frein_m"] == pytest.approx(20.0)
    assert m["margin_m"] == pytest.approx(-10.0)
    assert m["is_safe"] == 0.0


def test_two_second_rule():
    assert two_second_rule_distance(13.89, gap_s=2.0) == pytest.approx(27.78)
