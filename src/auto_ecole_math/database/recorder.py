"""Enregistrement d'une séance d'analyse en base de données.

``SessionRecorder`` fait le pont entre le pipeline (qui produit des lignes de
rapport frame par frame) et le schéma relationnel. Il est conçu pour être
utilisé comme gestionnaire de contexte::

    with SessionRecorder(video_path="seance.mov", student_id=1) as rec:
        for row in rows:
            rec.add_row(row)
        rec.finalize(stats_summary)

Les détections sont insérées par lots (``batch_size``) via l'API *bulk* de
SQLAlchemy : une vidéo d'une heure génère des centaines de milliers de lignes,
et un ``session.add()`` par ligne serait trop lent.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy import insert

from auto_ecole_math.database.db import get_session, init_db
from auto_ecole_math.database.models import (
    Calibration,
    Detection,
    Session,
    Statistics,
)

#: Seuil (s) en dessous duquel un TTC est considéré comme critique
CRITICAL_TTC_S = 3.0

#: Correspondance colonne CSV → colonne SQL
ROW_TO_COLUMN = {
    "Frame": "frame",
    "track_id": "track_id",
    "Categorie": "category",
    "Objet": "class_name",
    "Confiance": "confidence",
    "Confiance_lissee": "confidence_ewma",
    "Position_X_px": "position_x_px",
    "X_m": "x_m",
    "Y_m": "y_m",
    "vx_mps": "vx_mps",
    "vy_mps": "vy_mps",
    "speed_mps": "speed_mps",
    "d_m": "distance_m",
    "lateral_m": "lateral_m",
    "ttc_s": "ttc_s",
    "d_frein_m": "braking_distance_m",
    "margin_m": "margin_m",
}


def clean_float(value: Any) -> float | None:
    """Convertit une valeur de rapport en float stockable, sinon ``None``.

    Le pipeline écrit ``""`` pour un TTC infini et ``NaN`` pour une statistique
    non calculable : ni l'un ni l'autre n'a de sens en base, on stocke ``NULL``.
    """
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def clean_int(value: Any) -> int | None:
    """Convertit une valeur en entier stockable, sinon ``None``."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_to_detection(row: dict[str, Any], session_id: int) -> dict[str, Any]:
    """Traduit une ligne de rapport du pipeline en enregistrement ``Detection``."""
    payload: dict[str, Any] = {"session_id": session_id}
    for row_key, column in ROW_TO_COLUMN.items():
        value = row.get(row_key)
        if column in ("frame", "track_id"):
            payload[column] = clean_int(value)
        elif column in ("category", "class_name"):
            payload[column] = str(value) if value is not None else ""
        else:
            payload[column] = clean_float(value)
    # ``confidence`` est NOT NULL : valeur de repli si le pipeline n'a rien fourni
    if payload.get("confidence") is None:
        payload["confidence"] = 0.0
    return payload


def register_calibration(
    db, calibration_path: str | Path | None, *, name: str | None = None
) -> Calibration | None:
    """Enregistre (ou retrouve) la calibration utilisée pour une séance.

    Les calibrations identiques sont dédupliquées par empreinte SHA-256 du
    fichier : relancer dix analyses avec la même config ne crée qu'une ligne.
    """
    if calibration_path is None:
        return None
    path = Path(calibration_path)
    if not path.exists():
        return None

    config_json = path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    existing = db.query(Calibration).filter(Calibration.checksum == checksum).first()
    if existing is not None:
        return existing

    calibration = Calibration(
        name=name or path.name,
        config_json=config_json,
        checksum=checksum,
    )
    db.add(calibration)
    db.flush()
    return calibration


class SessionRecorder:
    """Persiste une séance d'analyse (séance + détections + statistiques)."""

    def __init__(
        self,
        *,
        video_path: str,
        student_id: int | None = None,
        instructor_id: int | None = None,
        calibration_path: str | Path | None = None,
        model_version: str | None = None,
        label: str | None = None,
        fps: float | None = None,
        reaction_time_s: float | None = None,
        deceleration_mps2: float | None = None,
        ego_speed_mps: float | None = None,
        db_url: str | None = None,
        batch_size: int = 500,
    ) -> None:
        self.db_url = db_url
        self.batch_size = max(1, batch_size)
        self._buffer: list[dict[str, Any]] = []
        self._n_detections = 0
        self._n_critical_ttc = 0
        self._n_negative_margin = 0
        self._max_frame = 0
        self._closed = False

        init_db(db_url)
        self.db = get_session(db_url)

        calibration = register_calibration(self.db, calibration_path)
        self.session = Session(
            student_id=student_id,
            instructor_id=instructor_id,
            calibration_id=calibration.id if calibration is not None else None,
            label=label,
            video_path=str(video_path),
            model_version=model_version,
            fps=clean_float(fps),
            reaction_time_s=clean_float(reaction_time_s),
            deceleration_mps2=clean_float(deceleration_mps2),
            ego_speed_mps=clean_float(ego_speed_mps),
        )
        self.db.add(self.session)
        self.db.flush()  # attribue session.id

    # -- Contexte -------------------------------------------------------

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    # -- Écriture -------------------------------------------------------

    @property
    def session_id(self) -> int:
        """Identifiant de la séance en base."""
        return self.session.id

    def add_row(self, row: dict[str, Any]) -> None:
        """Bufferise une ligne de rapport ; écrit dès que le lot est plein."""
        payload = row_to_detection(row, self.session.id)
        self._buffer.append(payload)
        self._n_detections += 1

        frame = payload.get("frame")
        if frame is not None and frame > self._max_frame:
            self._max_frame = frame

        ttc = payload.get("ttc_s")
        if ttc is not None and ttc < CRITICAL_TTC_S:
            self._n_critical_ttc += 1
        margin = payload.get("margin_m")
        if margin is not None and margin < 0:
            self._n_negative_margin += 1

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def add_rows(self, rows: list[dict[str, Any]]) -> None:
        """Bufferise plusieurs lignes."""
        for row in rows:
            self.add_row(row)

    def flush(self) -> None:
        """Écrit le buffer courant en base (insertion groupée)."""
        if not self._buffer:
            return
        self.db.execute(insert(Detection), self._buffer)
        self._buffer.clear()

    # -- Clôture --------------------------------------------------------

    def finalize(
        self,
        summary: dict[str, Any] | None = None,
        *,
        n_frames: int | None = None,
        duration_s: float | None = None,
    ) -> int:
        """Écrit les statistiques de séance et valide la transaction.

        Retourne l'identifiant de la séance créée.
        """
        self.flush()

        self.session.n_frames = n_frames if n_frames is not None else self._max_frame
        if duration_s is not None:
            self.session.duration_s = clean_float(duration_s)
        elif self.session.fps and self.session.n_frames:
            self.session.duration_s = self.session.n_frames / self.session.fps

        summary = summary or {}
        histogram = summary.get("ttc_histogram")
        stats = Statistics(
            session_id=self.session.id,
            distance_mean=clean_float(summary.get("distance_m_mean")),
            distance_std=clean_float(summary.get("distance_m_std")),
            distance_min=clean_float(summary.get("distance_m_min")),
            distance_max=clean_float(summary.get("distance_m_max")),
            distance_p50=clean_float(summary.get("distance_m_p50")),
            distance_p90=clean_float(summary.get("distance_m_p90")),
            lateral_mean=clean_float(summary.get("lateral_m_mean")),
            lateral_std=clean_float(summary.get("lateral_m_std")),
            ttc_mean=clean_float(summary.get("ttc_s_mean")),
            ttc_std=clean_float(summary.get("ttc_s_std")),
            ttc_min=clean_float(summary.get("ttc_s_min")),
            ttc_p50=clean_float(summary.get("ttc_s_p50")),
            ttc_p90=clean_float(summary.get("ttc_s_p90")),
            speed_mean=clean_float(summary.get("speed_mps_mean")),
            speed_max=clean_float(summary.get("speed_mps_max")),
            confidence_mean=clean_float(summary.get("confidence_mean")),
            confidence_std=clean_float(summary.get("confidence_std")),
            n_events=clean_int(summary.get("n_events")) or self._n_detections,
            n_critical_ttc=self._n_critical_ttc,
            n_negative_margin=self._n_negative_margin,
            ttc_histogram_json=(
                json.dumps(histogram, ensure_ascii=False) if histogram else None
            ),
        )
        self.db.add(stats)
        self.db.commit()
        session_id = self.session.id
        self._closed = True
        self.db.close()
        return session_id

    def close(self) -> None:
        """Valide ce qui a été écrit sans statistiques (clôture minimale)."""
        if self._closed:
            return
        self.flush()
        self.db.commit()
        self._closed = True
        self.db.close()

    def abort(self) -> None:
        """Annule la séance en cours (rollback complet)."""
        if self._closed:
            return
        self._buffer.clear()
        self.db.rollback()
        self._closed = True
        self.db.close()
