"""Tests — lissage de confiance et stats de séance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.uncertainty.confidence import (  # noqa: E402
    BetaBernoulliConfidence,
    ExponentialSmoother,
)
from auto_ecole_math.uncertainty.session_stats import SessionStatsCollector  # noqa: E402


def test_ewma():
    s = ExponentialSmoother(alpha=0.5)
    assert s.update(1.0) == pytest.approx(1.0)
    assert s.update(0.0) == pytest.approx(0.5)


def test_beta_bernoulli():
    bb = BetaBernoulliConfidence(alpha0=1, beta0=1, threshold=0.5)
    for _ in range(8):
        bb.update(0.9)
    for _ in range(2):
        bb.update(0.1)
    summary = bb.summary()
    assert summary["posterior_mean"] == pytest.approx(9 / 12)
    assert summary["posterior_var"] > 0


def test_session_stats(tmp_path):
    col = SessionStatsCollector()
    for d in [10.0, 12.0, 14.0]:
        col.add(distance_m=d, ttc_s=2.0, lateral_m=0.5, speed_mps=5.0, confidence=0.8)
    report = col.summarize()
    assert report["distance_m_mean"] == pytest.approx(12.0)
    assert report["distance_m_count"] == 3.0
    out = tmp_path / "stats.json"
    col.save_json(out)
    assert out.exists()
