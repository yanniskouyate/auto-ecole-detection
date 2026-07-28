"""Suivi multi-objets avec association et filtre de Kalman CV.

Pipeline par frame :

1. Projection des détections YOLO (pied de bbox) vers le plan sol (m).
2. Association Hongroise (ou greedy) via distance de Mahalanobis / Euclidienne.
3. ``predict`` Kalman pour toutes les pistes, ``update`` pour les associées.
4. Coasting pendant occlusions jusqu'à ``max_age`` frames sans mesure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanCV2D


@dataclass
class Detection:
    """Détection projetée dans le plan sol."""

    xy: np.ndarray  # (X, Y) mètres
    label: str
    confidence: float
    xyxy: np.ndarray | None = None  # bbox pixels optionnelle
    frame: int = 0


@dataclass
class Track:
    """Piste suivie avec état Kalman et métadonnées métier."""

    track_id: int
    label: str
    kf: KalmanCV2D
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confidence: float = 0.0
    last_xyxy: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed


class MultiObjectTracker:
    """Tracker multi-objets Kalman + association Hongroise.

    Parameters
    ----------
    max_age :
        Nombre max de frames en coasting (occlusion) avant suppression.
    min_hits :
        Nombre minimal de hits avant de considérer la piste confirmée.
    gate :
        Seuil de gating (mètres pour distance euclidienne, ou Mahalanobis).
    use_mahalanobis :
        Si ``True``, coût = distance de Mahalanobis ; sinon Euclidienne.
    dt :
        Pas de temps nominal (s).
    process_var / meas_var :
        Hyperparamètres du filtre de Kalman.
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 2,
        gate: float = 5.0,
        use_mahalanobis: bool = True,
        dt: float = 1.0 / 30.0,
        process_var: float = 1.0,
        meas_var: float = 0.5,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate = gate
        self.use_mahalanobis = use_mahalanobis
        self.dt = dt
        self.process_var = process_var
        self.meas_var = meas_var
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanCV2D(dt=self.dt, process_var=self.process_var, meas_var=self.meas_var)
        kf.initiate(det.xy)
        track = Track(
            track_id=self._next_id,
            label=det.label,
            kf=kf,
            confidence=det.confidence,
            last_xyxy=None if det.xyxy is None else np.asarray(det.xyxy, dtype=np.float64),
            history=[kf.position.copy()],
        )
        self._next_id += 1
        return track

    def _cost_matrix(self, detections: Sequence[Detection]) -> np.ndarray:
        n_t, n_d = len(self.tracks), len(detections)
        cost = np.full((n_t, n_d), fill_value=1e6, dtype=np.float64)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Prefer same class when possible by large penalty
                class_penalty = 0.0 if track.label == det.label else self.gate * 2.0
                if self.use_mahalanobis:
                    d = track.kf.mahalanobis(det.xy)
                else:
                    d = float(np.linalg.norm(track.kf.position - det.xy))
                cost[i, j] = d + class_penalty
        return cost

    def update(
        self,
        detections: Sequence[Detection],
        dt: float | None = None,
    ) -> list[Track]:
        """Met à jour le tracker avec les détections de la frame courante.

        Returns
        -------
        list[Track]
            Pistes confirmées (:math:`\\mathrm{hits} \\ge \\mathrm{min\\_hits}`
            ou récemment mises à jour).
        """
        self.frame_count += 1
        step_dt = self.dt if dt is None else float(dt)

        for track in self.tracks:
            track.kf.predict(dt=step_dt)
            track.age += 1
            track.time_since_update += 1

        dets = list(detections)
        if self.tracks and dets:
            cost = self._cost_matrix(dets)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > self.gate:
                    continue
                track = self.tracks[r]
                det = dets[c]
                track.kf.update(det.xy)
                track.hits += 1
                track.time_since_update = 0
                track.confidence = det.confidence
                track.label = det.label
                if det.xyxy is not None:
                    track.last_xyxy = np.asarray(det.xyxy, dtype=np.float64)
                track.history.append(track.kf.position.copy())
                matched_t.add(r)
                matched_d.add(c)

            unmatched_d = [j for j in range(len(dets)) if j not in matched_d]
        else:
            unmatched_d = list(range(len(dets)))
            matched_t = set()

        for j in unmatched_d:
            self.tracks.append(self._new_track(dets[j]))

        # Remove stale tracks
        alive: list[Track] = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                if track.time_since_update > 0:
                    # coasting: keep predicted position in history
                    track.history.append(track.kf.position.copy())
                alive.append(track)
        self.tracks = alive

        return self.confirmed_tracks()

    def confirmed_tracks(self) -> list[Track]:
        """Retourne les pistes confirmées (assez de hits ou déjà stables)."""
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits or (t.age >= self.min_hits and t.time_since_update == 0):
                out.append(t)
        return out

    def to_dicts(self) -> list[dict[str, Any]]:
        """Sérialise les pistes confirmées pour logging / CSV."""
        rows = []
        for t in self.confirmed_tracks():
            rows.append(
                {
                    "track_id": t.track_id,
                    "label": t.label,
                    "confidence": t.confidence,
                    "X_m": float(t.position[0]),
                    "Y_m": float(t.position[1]),
                    "vx_mps": float(t.velocity[0]),
                    "vy_mps": float(t.velocity[1]),
                    "speed_mps": t.speed,
                    "hits": t.hits,
                    "time_since_update": t.time_since_update,
                }
            )
        return rows
