"""Initialise la base de données (création des tables).

Usage::

    PYTHONPATH=src python scripts/init_db.py
    PYTHONPATH=src python scripts/init_db.py --demo      # + élève/moniteur d'exemple
    PYTHONPATH=src python scripts/init_db.py --reset     # DESTRUCTIF : vide tout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.database.db import (  # noqa: E402
    default_db_url,
    init_db,
    reset_db,
    session_scope,
)
from auto_ecole_math.database.queries import (  # noqa: E402
    database_overview,
    get_or_create_instructor,
    get_or_create_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialisation de la base auto-école")
    parser.add_argument("--db-url", default=None, help="URL SQLAlchemy (défaut : SQLite local)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime et recrée toutes les tables (perte de données)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Crée un élève et un moniteur de démonstration",
    )
    args = parser.parse_args()

    url = args.db_url or default_db_url()

    if args.reset:
        confirm = input(f"Vider TOUTES les tables de {url} ? [oui/N] ").strip().lower()
        if confirm != "oui":
            print("Annulé.")
            return
        reset_db(url)
        print("Tables réinitialisées.")
    else:
        init_db(url, verbose=True)

    if args.demo:
        with session_scope(url) as db:
            student = get_or_create_student(db, "Élève Démo", email="eleve@demo.fr")
            instructor = get_or_create_instructor(db, "Moniteur Démo", email="moniteur@demo.fr")
            print(f"Élève #{student.id} — {student.name}")
            print(f"Moniteur #{instructor.id} — {instructor.name}")

    print("Contenu :", database_overview(url))


if __name__ == "__main__":
    main()
