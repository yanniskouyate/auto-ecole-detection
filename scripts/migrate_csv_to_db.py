"""Importe un rapport CSV existant (+ stats JSON) dans la base.

Permet de ne pas perdre les analyses produites avant la mise en place de la
persistance.

Usage::

    PYTHONPATH=src python scripts/migrate_csv_to_db.py
    PYTHONPATH=src python scripts/migrate_csv_to_db.py \\
        --csv rapport_conduite.csv --stats rapport_stats.json \\
        --student "Alice Dupont" --label "Séance du 12 mars"
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.database.db import session_scope  # noqa: E402
from auto_ecole_math.database.queries import get_or_create_student  # noqa: E402
from auto_ecole_math.database.recorder import SessionRecorder  # noqa: E402


def load_stats(path: Path | None) -> dict:
    """Charge le résumé statistique, ou un dict vide s'il est absent."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migration CSV → base de données")
    parser.add_argument("--csv", default=str(ROOT / "rapport_conduite.csv"))
    parser.add_argument("--stats", default=str(ROOT / "rapport_stats.json"))
    parser.add_argument("--video", default="auto_ecole_test.mov")
    parser.add_argument("--calib", default=str(ROOT / "config" / "homography_default.json"))
    parser.add_argument("--model", default="yolo11n")
    parser.add_argument("--student", default=None, help="Nom de l'élève (créé si absent)")
    parser.add_argument("--label", default=None, help="Libellé de la séance")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV introuvable : {csv_path}")

    stats_path = Path(args.stats) if args.stats else None
    summary = load_stats(stats_path)

    student_id = None
    if args.student:
        with session_scope(args.db_url) as db:
            student_id = get_or_create_student(db, args.student).id

    label = args.label or f"Import {csv_path.name}"

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))

    if not rows:
        raise SystemExit("Le CSV ne contient aucune ligne.")

    recorder = SessionRecorder(
        video_path=args.video,
        student_id=student_id,
        calibration_path=args.calib,
        model_version=args.model,
        label=label,
        db_url=args.db_url,
    )
    try:
        recorder.add_rows(rows)
        session_id = recorder.finalize(summary)
    except BaseException:
        recorder.abort()
        raise

    print(f"Import terminé : {len(rows)} détections → séance #{session_id}")


if __name__ == "__main__":
    main()
