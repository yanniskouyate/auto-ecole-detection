"""Connexion et gestion de la base de données.

Par défaut la base est un fichier SQLite à la racine du dépôt
(``auto_ecole.db``). L'URL est surchargeable par la variable d'environnement
``AUTO_ECOLE_DB_URL``, ce qui permet de basculer vers PostgreSQL sans toucher
au code::

    export AUTO_ECOLE_DB_URL="postgresql+psycopg://user:pwd@localhost/auto_ecole"

Usage courant::

    from auto_ecole_math.database import init_db, session_scope

    init_db()
    with session_scope() as db:
        db.add(Student(name="Alice"))
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from auto_ecole_math.database.models import Base

#: Racine du dépôt (``src/auto_ecole_math/database/db.py`` → 3 niveaux au-dessus)
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Chemin par défaut du fichier SQLite
DEFAULT_DB_PATH = REPO_ROOT / "auto_ecole.db"

ENV_VAR = "AUTO_ECOLE_DB_URL"

_engines: dict[str, Engine] = {}
_sessionmakers: dict[str, sessionmaker[OrmSession]] = {}


def default_db_url() -> str:
    """URL de connexion par défaut (env var sinon SQLite local)."""
    return os.environ.get(ENV_VAR) or f"sqlite:///{DEFAULT_DB_PATH}"


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Active les clés étrangères sur SQLite (désactivées par défaut).

    Sans ce PRAGMA, ``ON DELETE CASCADE`` est ignoré et supprimer une séance
    laisserait ses détections orphelines.
    """
    module = type(dbapi_connection).__module__
    if not module.startswith("sqlite3"):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Retourne (et mémorise) le moteur SQLAlchemy pour une URL donnée."""
    url = url or default_db_url()
    engine = _engines.get(url)
    if engine is None:
        connect_args = {}
        if url.startswith("sqlite"):
            # Autorise l'usage depuis les threads Streamlit
            connect_args["check_same_thread"] = False
        engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        _engines[url] = engine
    return engine


def get_sessionmaker(url: str | None = None) -> sessionmaker[OrmSession]:
    """Retourne (et mémorise) la fabrique de sessions pour une URL donnée."""
    url = url or default_db_url()
    maker = _sessionmakers.get(url)
    if maker is None:
        maker = sessionmaker(bind=get_engine(url), expire_on_commit=False)
        _sessionmakers[url] = maker
    return maker


def init_db(url: str | None = None, *, verbose: bool = False) -> Engine:
    """Crée les tables manquantes et retourne le moteur.

    Idempotent : peut être appelé à chaque démarrage sans risque.
    """
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    if verbose:
        print(f"Base de données prête : {engine.url}")
    return engine


def get_session(url: str | None = None) -> OrmSession:
    """Ouvre une session SQLAlchemy (à fermer manuellement)."""
    return get_sessionmaker(url)()


@contextmanager
def session_scope(url: str | None = None) -> Iterator[OrmSession]:
    """Session transactionnelle : commit en sortie, rollback si exception."""
    session = get_session(url)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db(url: str | None = None) -> None:
    """Supprime puis recrée toutes les tables. **Destructif** (tests / dev)."""
    engine = get_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def dispose_engines() -> None:
    """Ferme tous les moteurs mémorisés (utile en fin de tests)."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _sessionmakers.clear()
