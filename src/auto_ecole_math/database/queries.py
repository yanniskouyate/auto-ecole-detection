"""Requêtes prêtes à l'emploi sur l'historique des séances.

Ces fonctions encapsulent les questions métier courantes (« quelles séances
pour cet élève ? », « où l'élève s'est-il approché dangereusement ? ») pour que
l'interface n'ait pas à écrire de SQL.
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import joinedload

from auto_ecole_math.database.db import session_scope
from auto_ecole_math.database.models import (
    Detection,
    Instructor,
    Session,
    Statistics,
    Student,
)
from auto_ecole_math.database.recorder import CRITICAL_TTC_S


def list_sessions(
    db: OrmSession, *, student_id: int | None = None, limit: int = 100
) -> list[Session]:
    """Séances les plus récentes, éventuellement filtrées par élève."""
    stmt = (
        select(Session)
        .options(
            joinedload(Session.student),
            joinedload(Session.instructor),
            joinedload(Session.statistics),
        )
        .order_by(Session.date.desc())
        .limit(limit)
    )
    if student_id is not None:
        stmt = stmt.where(Session.student_id == student_id)
    return list(db.scalars(stmt).unique())


def get_session_by_id(db: OrmSession, session_id: int) -> Session | None:
    """Une séance avec son élève, son moniteur et ses statistiques."""
    stmt = (
        select(Session)
        .options(
            joinedload(Session.student),
            joinedload(Session.instructor),
            joinedload(Session.statistics),
            joinedload(Session.calibration),
        )
        .where(Session.id == session_id)
    )
    return db.scalars(stmt).unique().one_or_none()


def get_statistics(db: OrmSession, session_id: int) -> Statistics | None:
    """Statistiques agrégées d'une séance."""
    return db.scalars(
        select(Statistics).where(Statistics.session_id == session_id)
    ).one_or_none()


def session_detections(
    db: OrmSession,
    session_id: int,
    *,
    categories: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    max_distance_m: float | None = None,
    max_ttc_s: float | None = None,
) -> list[Detection]:
    """Détections d'une séance, avec filtres métier optionnels."""
    stmt = select(Detection).where(Detection.session_id == session_id)
    if categories:
        stmt = stmt.where(Detection.category.in_(list(categories)))
    if classes:
        stmt = stmt.where(Detection.class_name.in_(list(classes)))
    if max_distance_m is not None:
        stmt = stmt.where(Detection.distance_m <= max_distance_m)
    if max_ttc_s is not None:
        stmt = stmt.where(
            Detection.ttc_s.is_not(None), Detection.ttc_s <= max_ttc_s
        )
    stmt = stmt.order_by(Detection.frame, Detection.track_id)
    return list(db.scalars(stmt))


def critical_events(
    db: OrmSession,
    session_id: int,
    *,
    ttc_threshold_s: float = CRITICAL_TTC_S,
) -> list[Detection]:
    """Événements à risque : TTC sous le seuil **ou** marge de freinage négative."""
    stmt = (
        select(Detection)
        .where(Detection.session_id == session_id)
        .where(
            (
                Detection.ttc_s.is_not(None) & (Detection.ttc_s < ttc_threshold_s)
            )
            | (Detection.margin_m.is_not(None) & (Detection.margin_m < 0))
        )
        .order_by(Detection.frame)
    )
    return list(db.scalars(stmt))


def vulnerable_users_close(
    db: OrmSession, session_id: int, *, max_distance_m: float = 5.0
) -> list[Detection]:
    """Piétons et cyclistes détectés à moins de ``max_distance_m``."""
    stmt = (
        select(Detection)
        .where(Detection.session_id == session_id)
        .where(Detection.category == "Usager Vulnérable")
        .where(Detection.distance_m.is_not(None), Detection.distance_m < max_distance_m)
        .order_by(Detection.distance_m)
    )
    return list(db.scalars(stmt))


def student_progress(db: OrmSession, student_id: int) -> list[dict[str, Any]]:
    """Évolution des indicateurs clés d'un élève, séance par séance."""
    stmt = (
        select(Session, Statistics)
        .join(Statistics, Statistics.session_id == Session.id)
        .where(Session.student_id == student_id)
        .order_by(Session.date)
    )
    rows = []
    for session, stats in db.execute(stmt):
        rows.append(
            {
                "session_id": session.id,
                "date": session.date,
                "label": session.label,
                "distance_mean": stats.distance_mean,
                "ttc_p50": stats.ttc_p50,
                "n_events": stats.n_events,
                "n_critical_ttc": stats.n_critical_ttc,
                "n_negative_margin": stats.n_negative_margin,
            }
        )
    return rows


def ttc_by_student(db: OrmSession) -> list[dict[str, Any]]:
    """TTC moyen et nombre d'événements critiques, agrégés par élève."""
    stmt = (
        select(
            Student.id,
            Student.name,
            func.count(func.distinct(Session.id)).label("n_sessions"),
            func.avg(Detection.ttc_s).label("avg_ttc"),
            func.min(Detection.distance_m).label("min_distance"),
        )
        .join(Session, Session.student_id == Student.id)
        .join(Detection, Detection.session_id == Session.id)
        .group_by(Student.id)
        .order_by(Student.name)
    )
    return [
        {
            "student_id": sid,
            "name": name,
            "n_sessions": n_sessions,
            "avg_ttc_s": avg_ttc,
            "min_distance_m": min_distance,
        }
        for sid, name, n_sessions, avg_ttc, min_distance in db.execute(stmt)
    ]


def detections_dataframe(db: OrmSession, session_id: int):
    """Détections d'une séance sous forme de ``pandas.DataFrame``.

    Les colonnes reprennent les noms du CSV historique pour que l'interface
    existante fonctionne sans modification.
    """
    import pandas as pd

    stmt = (
        select(Detection)
        .where(Detection.session_id == session_id)
        .order_by(Detection.frame, Detection.track_id)
    )
    rows = [
        {
            "Frame": d.frame,
            "track_id": d.track_id,
            "Categorie": d.category,
            "Objet": d.class_name,
            "Confiance": d.confidence,
            "Confiance_lissee": d.confidence_ewma,
            "Position_X_px": d.position_x_px,
            "X_m": d.x_m,
            "Y_m": d.y_m,
            "vx_mps": d.vx_mps,
            "vy_mps": d.vy_mps,
            "speed_mps": d.speed_mps,
            "d_m": d.distance_m,
            "lateral_m": d.lateral_m,
            "ttc_s": d.ttc_s,
            "d_frein_m": d.braking_distance_m,
            "margin_m": d.margin_m,
        }
        for d in db.scalars(stmt)
    ]
    return pd.DataFrame(rows)


def statistics_as_summary(stats: Statistics | None) -> dict[str, Any]:
    """Reconstruit un dictionnaire au format ``rapport_stats.json``.

    Permet de réutiliser tel quel l'affichage Streamlit écrit pour le JSON.
    """
    import json

    if stats is None:
        return {}
    histogram = {}
    if stats.ttc_histogram_json:
        try:
            histogram = json.loads(stats.ttc_histogram_json)
        except json.JSONDecodeError:
            histogram = {}
    return {
        "distance_m_mean": stats.distance_mean,
        "distance_m_std": stats.distance_std,
        "distance_m_min": stats.distance_min,
        "distance_m_max": stats.distance_max,
        "distance_m_p50": stats.distance_p50,
        "distance_m_p90": stats.distance_p90,
        "lateral_m_mean": stats.lateral_mean,
        "lateral_m_std": stats.lateral_std,
        "ttc_s_mean": stats.ttc_mean,
        "ttc_s_std": stats.ttc_std,
        "ttc_s_min": stats.ttc_min,
        "ttc_s_p50": stats.ttc_p50,
        "ttc_s_p90": stats.ttc_p90,
        "speed_mps_mean": stats.speed_mean,
        "speed_mps_max": stats.speed_max,
        "confidence_mean": stats.confidence_mean,
        "confidence_std": stats.confidence_std,
        "n_events": stats.n_events,
        "n_critical_ttc": stats.n_critical_ttc,
        "n_negative_margin": stats.n_negative_margin,
        "ttc_histogram": histogram,
    }


def delete_session(db: OrmSession, session_id: int) -> bool:
    """Supprime une séance et, en cascade, ses détections et statistiques."""
    session = db.get(Session, session_id)
    if session is None:
        return False
    db.delete(session)
    return True


def get_or_create_student(
    db: OrmSession, name: str, *, email: str | None = None
) -> Student:
    """Retrouve un élève par nom (ou email), le crée sinon."""
    stmt = select(Student)
    stmt = stmt.where(Student.email == email) if email else stmt.where(
        Student.name == name
    )
    student = db.scalars(stmt).first()
    if student is None:
        student = Student(name=name, email=email)
        db.add(student)
        db.flush()
    return student


def get_or_create_instructor(
    db: OrmSession, name: str, *, email: str | None = None
) -> Instructor:
    """Retrouve un moniteur par nom (ou email), le crée sinon."""
    stmt = select(Instructor)
    stmt = stmt.where(Instructor.email == email) if email else stmt.where(
        Instructor.name == name
    )
    instructor = db.scalars(stmt).first()
    if instructor is None:
        instructor = Instructor(name=name, email=email)
        db.add(instructor)
        db.flush()
    return instructor


def database_overview(db_url: str | None = None) -> dict[str, int]:
    """Compte les enregistrements de chaque table (diagnostic rapide)."""
    with session_scope(db_url) as db:
        return {
            "students": db.scalar(select(func.count()).select_from(Student)) or 0,
            "instructors": db.scalar(select(func.count()).select_from(Instructor)) or 0,
            "sessions": db.scalar(select(func.count()).select_from(Session)) or 0,
            "detections": db.scalar(select(func.count()).select_from(Detection)) or 0,
        }
