"""Modèles SQLAlchemy pour la persistance des séances de conduite.

Schéma relationnel::

    students ──────┐
                   ├──< sessions >──┬──< detections
    instructors ───┤                └─── statistics (1–1)
                   │
    calibrations ──┘

Une ``Session`` est une séance de conduite analysée (une vidéo). Elle porte
toutes les ``Detection`` (une ligne par objet suivi et par frame) et un unique
enregistrement ``Statistics`` (agrégats de la séance).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Horodatage UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base déclarative commune à tous les modèles."""


class Student(Base):
    """Élève conducteur."""

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    phone: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list[Session]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Student id={self.id} name={self.name!r}>"


class Instructor(Base):
    """Moniteur d'auto-école."""

    __tablename__ = "instructors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list[Session]] = relationship(back_populates="instructor")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Instructor id={self.id} name={self.name!r}>"


class Calibration(Base):
    """Jeu de paramètres d'homographie utilisé pour une séance.

    ``config_json`` contient le contenu brut de ``config/homography_default.json``
    afin de garantir la traçabilité : on sait exactement avec quelle calibration
    une mesure en mètres a été produite.
    """

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(64), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list[Session]] = relationship(back_populates="calibration")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Calibration id={self.id} name={self.name!r}>"


class Session(Base):
    """Une séance de conduite analysée (une vidéo traitée par le pipeline)."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int | None] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), index=True
    )
    instructor_id: Mapped[int | None] = mapped_column(
        ForeignKey("instructors.id"), index=True
    )
    calibration_id: Mapped[int | None] = mapped_column(ForeignKey("calibrations.id"))

    label: Mapped[str | None] = mapped_column(String(255))
    date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))

    fps: Mapped[float | None] = mapped_column(Float)
    duration_s: Mapped[float | None] = mapped_column(Float)
    n_frames: Mapped[int | None] = mapped_column(Integer)

    # Paramètres physiques du modèle de freinage (traçabilité)
    reaction_time_s: Mapped[float | None] = mapped_column(Float)
    deceleration_mps2: Mapped[float | None] = mapped_column(Float)
    ego_speed_mps: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    student: Mapped[Student | None] = relationship(back_populates="sessions")
    instructor: Mapped[Instructor | None] = relationship(back_populates="sessions")
    calibration: Mapped[Calibration | None] = relationship(back_populates="sessions")
    detections: Mapped[list[Detection]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    statistics: Mapped[Statistics | None] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Session id={self.id} video={self.video_path!r} date={self.date}>"


class Detection(Base):
    """Un objet suivi (track) observé sur une frame donnée.

    C'est la table volumineuse : une vidéo d'une heure à 30 fps avec 5 objets
    simultanés produit ~540 000 lignes. Les insertions passent donc par
    ``bulk_insert`` (voir ``recorder.py``).
    """

    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_session_frame", "session_id", "frame"),
        Index("ix_detections_session_track", "session_id", "track_id"),
        Index("ix_detections_session_class", "session_id", "class_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    frame: Mapped[int] = mapped_column(Integer, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)

    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_ewma: Mapped[float | None] = mapped_column(Float)

    position_x_px: Mapped[float | None] = mapped_column(Float)

    # État filtré (Kalman) dans le plan sol, en mètres
    x_m: Mapped[float | None] = mapped_column(Float)
    y_m: Mapped[float | None] = mapped_column(Float)
    vx_mps: Mapped[float | None] = mapped_column(Float)
    vy_mps: Mapped[float | None] = mapped_column(Float)
    speed_mps: Mapped[float | None] = mapped_column(Float)

    # Métriques de sécurité
    distance_m: Mapped[float | None] = mapped_column(Float)
    lateral_m: Mapped[float | None] = mapped_column(Float)
    ttc_s: Mapped[float | None] = mapped_column(Float, index=True)
    braking_distance_m: Mapped[float | None] = mapped_column(Float)
    margin_m: Mapped[float | None] = mapped_column(Float)

    session: Mapped[Session] = relationship(back_populates="detections")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Detection session={self.session_id} frame={self.frame} "
            f"track={self.track_id} {self.class_name!r}>"
        )


class Statistics(Base):
    """Agrégats d'une séance (équivalent de ``rapport_stats.json``)."""

    __tablename__ = "statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    distance_mean: Mapped[float | None] = mapped_column(Float)
    distance_std: Mapped[float | None] = mapped_column(Float)
    distance_min: Mapped[float | None] = mapped_column(Float)
    distance_max: Mapped[float | None] = mapped_column(Float)
    distance_p50: Mapped[float | None] = mapped_column(Float)
    distance_p90: Mapped[float | None] = mapped_column(Float)

    lateral_mean: Mapped[float | None] = mapped_column(Float)
    lateral_std: Mapped[float | None] = mapped_column(Float)

    ttc_mean: Mapped[float | None] = mapped_column(Float)
    ttc_std: Mapped[float | None] = mapped_column(Float)
    ttc_min: Mapped[float | None] = mapped_column(Float)
    ttc_p50: Mapped[float | None] = mapped_column(Float)
    ttc_p90: Mapped[float | None] = mapped_column(Float)

    speed_mean: Mapped[float | None] = mapped_column(Float)
    speed_max: Mapped[float | None] = mapped_column(Float)

    confidence_mean: Mapped[float | None] = mapped_column(Float)
    confidence_std: Mapped[float | None] = mapped_column(Float)

    n_events: Mapped[int] = mapped_column(Integer, default=0)
    n_critical_ttc: Mapped[int] = mapped_column(Integer, default=0)
    n_negative_margin: Mapped[int] = mapped_column(Integer, default=0)

    # Histogramme TTC sérialisé (bins + counts)
    ttc_histogram_json: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="statistics")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Statistics session={self.session_id} n_events={self.n_events}>"
