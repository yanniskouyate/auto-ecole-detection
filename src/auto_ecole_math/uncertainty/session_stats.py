"""Statistiques de séance de conduite (agrégats temporels).

Calcule moyenne, variance et distributions empiriques sur les métriques
de sécurité collectées frame par frame :

.. math::

    \\bar d = \\frac{1}{N}\\sum_{i=1}^N d_i,\\qquad
    \\widehat{\\mathrm{Var}}(d) = \\frac{1}{N-1}\\sum_{i=1}^N (d_i - \\bar d)^2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _safe_array(values: Sequence[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


@dataclass
class SessionStatsCollector:
    """Accumulateur de métriques pour une séance vidéo."""

    distances: list[float] = field(default_factory=list)
    lateral_offsets: list[float] = field(default_factory=list)
    ttcs: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    reaction_proxies_s: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        distance_m: float | None = None,
        lateral_m: float | None = None,
        ttc_s: float | None = None,
        speed_mps: float | None = None,
        confidence: float | None = None,
        reaction_proxy_s: float | None = None,
        event: dict[str, Any] | None = None,
    ) -> None:
        if distance_m is not None and np.isfinite(distance_m):
            self.distances.append(float(distance_m))
        if lateral_m is not None and np.isfinite(lateral_m):
            self.lateral_offsets.append(float(lateral_m))
        if ttc_s is not None and np.isfinite(ttc_s) and ttc_s < 1e6:
            self.ttcs.append(float(ttc_s))
        if speed_mps is not None and np.isfinite(speed_mps):
            self.speeds.append(float(speed_mps))
        if confidence is not None and np.isfinite(confidence):
            self.confidences.append(float(confidence))
        if reaction_proxy_s is not None and np.isfinite(reaction_proxy_s):
            self.reaction_proxies_s.append(float(reaction_proxy_s))
        if event is not None:
            self.events.append(event)

    @staticmethod
    def _summarize(values: Sequence[float], prefix: str) -> dict[str, float]:
        arr = _safe_array(values)
        if arr.size == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_mean": float("nan"),
                f"{prefix}_var": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p90": float("nan"),
            }
        return {
            f"{prefix}_count": float(arr.size),
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_var": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
            f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_p50": float(np.percentile(arr, 50)),
            f"{prefix}_p90": float(np.percentile(arr, 90)),
        }

    def summarize(self) -> dict[str, Any]:
        """Agrégats statistiques de la séance."""
        report: dict[str, Any] = {}
        report.update(self._summarize(self.distances, "distance_m"))
        report.update(self._summarize(self.lateral_offsets, "lateral_m"))
        report.update(self._summarize(self.ttcs, "ttc_s"))
        report.update(self._summarize(self.speeds, "speed_mps"))
        report.update(self._summarize(self.confidences, "confidence"))
        report.update(self._summarize(self.reaction_proxies_s, "reaction_proxy_s"))
        report["n_events"] = len(self.events)
        # Empirical TTC histogram (fixed bins)
        ttc_arr = _safe_array(self.ttcs)
        if ttc_arr.size:
            bins = [0, 1, 2, 3, 5, 10, 20, 60]
            hist, edges = np.histogram(ttc_arr, bins=bins)
            report["ttc_histogram"] = {
                "bins": edges.tolist(),
                "counts": hist.tolist(),
            }
        else:
            report["ttc_histogram"] = {"bins": [], "counts": []}
        return report

    def save_json(self, path: str | Path) -> None:
        """Exporte le résumé JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.summarize(), f, indent=2, ensure_ascii=False)
            f.write("\n")
