"""Analyse métier auto-école — pipeline mathématique (homographie, Kalman, TTC).

Ce script est le point d'entrée CLI. La logique scientifique est dans
``src/auto_ecole_math/``.

Usage::

    PYTHONPATH=src python analyse_video.py
    PYTHONPATH=src python analyse_video.py --no-display
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.pipeline import DrivingAnalysisPipeline, PipelineConfig  # noqa: E402


def _resolve_people(args) -> tuple[int | None, int | None]:
    """Retrouve (ou crée) l'élève et le moniteur nommés en ligne de commande."""
    if args.no_db or (not args.student and not args.instructor):
        return None, None

    from auto_ecole_math.database.db import init_db, session_scope
    from auto_ecole_math.database.queries import (
        get_or_create_instructor,
        get_or_create_student,
    )

    init_db(args.db_url)
    student_id = instructor_id = None
    with session_scope(args.db_url) as db:
        if args.student:
            student_id = get_or_create_student(db, args.student).id
        if args.instructor:
            instructor_id = get_or_create_instructor(db, args.instructor).id
    return student_id, instructor_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse de conduite : YOLO + géométrie projective + Kalman + TTC"
    )
    parser.add_argument("--video", default=str(ROOT / "auto_ecole_test.mov"))
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument(
        "--calib",
        default=str(ROOT / "config" / "homography_default.json"),
    )
    parser.add_argument("--csv", default=str(ROOT / "rapport_conduite.csv"))
    parser.add_argument("--stats", default=str(ROOT / "rapport_stats.json"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--ego-speed", type=float, default=0.0, help="Vitesse ego (m/s), 0 si inconnue")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Désactive l'enregistrement en base (CSV + JSON uniquement)",
    )
    parser.add_argument("--db-url", default=None, help="URL SQLAlchemy (défaut : SQLite local)")
    parser.add_argument("--student", default=None, help="Nom de l'élève (créé si absent)")
    parser.add_argument("--instructor", default=None, help="Nom du moniteur (créé si absent)")
    parser.add_argument("--label", default=None, help="Libellé de la séance")
    args = parser.parse_args()

    student_id, instructor_id = _resolve_people(args)

    print("Chargement du pipeline mathématique...")
    cfg = PipelineConfig(
        video_path=args.video,
        model_path=args.model,
        calibration_path=args.calib,
        csv_path=args.csv,
        stats_path=args.stats,
        show_display=not args.no_display,
        ego_speed_mps=args.ego_speed,
        db_enabled=not args.no_db,
        db_url=args.db_url,
        student_id=student_id,
        instructor_id=instructor_id,
        session_label=args.label,
    )
    pipeline = DrivingAnalysisPipeline(cfg)
    stats = pipeline.run()
    summary = stats.summarize()
    print(
        f"Résumé séance — "
        f"d̄={summary.get('distance_m_mean', float('nan')):.2f} m, "
        f"TTC̄={summary.get('ttc_s_mean', float('nan')):.2f} s, "
        f"événements={summary.get('n_events', 0)}"
    )
    if pipeline.session_id is not None:
        print(f"Séance enregistrée en base sous l'identifiant #{pipeline.session_id}")


if __name__ == "__main__":
    main()
