"""Tests — persistance des séances (modèles, recorder, requêtes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.database.db import init_db, session_scope  # noqa: E402
from auto_ecole_math.database.models import Detection, Session  # noqa: E402
from auto_ecole_math.database.queries import (  # noqa: E402
    critical_events,
    database_overview,
    delete_session,
    detections_dataframe,
    get_or_create_student,
    get_session_by_id,
    get_statistics,
    list_sessions,
    statistics_as_summary,
    student_progress,
    vulnerable_users_close,
)
from auto_ecole_math.database.recorder import (  # noqa: E402
    SessionRecorder,
    clean_float,
    row_to_detection,
)


@pytest.fixture()
def db_url(tmp_path):
    """Base SQLite jetable, isolée pour chaque test."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    init_db(url)
    return url


def make_row(frame: int, **overrides):
    """Ligne de rapport au format produit par le pipeline."""
    row = {
        "Frame": frame,
        "track_id": 1,
        "Categorie": "Véhicule",
        "Objet": "car",
        "Confiance": 0.87,
        "Confiance_lissee": 0.86,
        "Position_X_px": 470.0,
        "X_m": 0.32,
        "Y_m": 5.43,
        "vx_mps": -0.001,
        "vy_mps": 0.04,
        "speed_mps": 0.04,
        "d_m": 5.44,
        "lateral_m": 0.32,
        "ttc_s": 12.0,
        "d_frein_m": 0.04,
        "margin_m": 5.4,
    }
    row.update(overrides)
    return row


# --- Nettoyage des valeurs -------------------------------------------------


def test_clean_float_rejette_non_finis():
    assert clean_float("") is None
    assert clean_float(None) is None
    assert clean_float(float("nan")) is None
    assert clean_float(float("inf")) is None
    assert clean_float("3.5") == pytest.approx(3.5)


def test_row_to_detection_mappe_les_colonnes():
    payload = row_to_detection(make_row(7, ttc_s=""), session_id=42)
    assert payload["session_id"] == 42
    assert payload["frame"] == 7
    assert payload["class_name"] == "car"
    assert payload["distance_m"] == pytest.approx(5.44)
    # TTC infini écrit "" par le pipeline → NULL en base
    assert payload["ttc_s"] is None


# --- Enregistrement --------------------------------------------------------


def test_recorder_persiste_seance_et_detections(db_url):
    with session_scope(db_url) as db:
        student_id = get_or_create_student(db, "Alice").id

    recorder = SessionRecorder(
        video_path="seance.mov",
        student_id=student_id,
        fps=30.0,
        db_url=db_url,
        batch_size=2,
    )
    recorder.add_rows([make_row(i) for i in range(1, 6)])
    session_id = recorder.finalize({"distance_m_mean": 5.4, "n_events": 5})

    with session_scope(db_url) as db:
        session = get_session_by_id(db, session_id)
        assert session is not None
        assert session.student.name == "Alice"
        assert session.n_frames == 5
        assert session.duration_s == pytest.approx(5 / 30.0)
        assert len(session.detections) == 5


def test_recorder_compte_les_evenements_critiques(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_row(make_row(1, ttc_s=1.2))  # TTC critique
    recorder.add_row(make_row(2, margin_m=-3.0))  # marge négative
    recorder.add_row(make_row(3))  # sain
    session_id = recorder.finalize({})

    with session_scope(db_url) as db:
        stats = get_statistics(db, session_id)
        assert stats.n_critical_ttc == 1
        assert stats.n_negative_margin == 1
        assert critical_events(db, session_id).__len__() == 2


def test_recorder_abort_ne_laisse_rien(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_rows([make_row(i) for i in range(1, 4)])
    recorder.abort()

    overview = database_overview(db_url)
    assert overview["sessions"] == 0
    assert overview["detections"] == 0


def test_recorder_comme_context_manager_rollback(db_url):
    with pytest.raises(RuntimeError):
        with SessionRecorder(video_path="s.mov", db_url=db_url) as rec:
            rec.add_row(make_row(1))
            raise RuntimeError("échec simulé")

    assert database_overview(db_url)["sessions"] == 0


def test_statistiques_nan_stockees_en_null(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_row(make_row(1))
    session_id = recorder.finalize({"ttc_s_mean": float("nan"), "distance_m_mean": 4.0})

    with session_scope(db_url) as db:
        stats = get_statistics(db, session_id)
        assert stats.ttc_mean is None
        assert stats.distance_mean == pytest.approx(4.0)


def test_calibration_dedupliquee(db_url, tmp_path):
    calib = tmp_path / "calib.json"
    calib.write_text('{"image_size": [1280, 832]}', encoding="utf-8")

    ids = []
    for _ in range(3):
        rec = SessionRecorder(
            video_path="s.mov", calibration_path=calib, db_url=db_url
        )
        rec.add_row(make_row(1))
        ids.append(rec.finalize({}))

    with session_scope(db_url) as db:
        calib_ids = {get_session_by_id(db, sid).calibration_id for sid in ids}
    # Même fichier de calibration → une seule ligne partagée
    assert len(calib_ids) == 1


# --- Requêtes --------------------------------------------------------------


def test_suppression_en_cascade(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_rows([make_row(i) for i in range(1, 11)])
    session_id = recorder.finalize({})

    with session_scope(db_url) as db:
        assert delete_session(db, session_id) is True

    overview = database_overview(db_url)
    assert overview["sessions"] == 0
    assert overview["detections"] == 0


def test_usagers_vulnerables_proches(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_row(make_row(1, Categorie="Usager Vulnérable", Objet="person", d_m=3.0))
    recorder.add_row(make_row(2, Categorie="Usager Vulnérable", Objet="person", d_m=20.0))
    recorder.add_row(make_row(3, d_m=2.0))  # véhicule, pas un usager vulnérable
    session_id = recorder.finalize({})

    with session_scope(db_url) as db:
        proches = vulnerable_users_close(db, session_id, max_distance_m=5.0)
        assert len(proches) == 1
        assert proches[0].distance_m == pytest.approx(3.0)


def test_progression_eleve_sur_plusieurs_seances(db_url):
    with session_scope(db_url) as db:
        student_id = get_or_create_student(db, "Bob").id

    for n, mean in enumerate([4.0, 6.0, 8.0], start=1):
        rec = SessionRecorder(
            video_path=f"s{n}.mov", student_id=student_id, db_url=db_url
        )
        rec.add_row(make_row(1))
        rec.finalize({"distance_m_mean": mean})

    with session_scope(db_url) as db:
        progression = student_progress(db, student_id)

    assert [p["distance_mean"] for p in progression] == pytest.approx([4.0, 6.0, 8.0])


def test_liste_seances_filtree_par_eleve(db_url):
    with session_scope(db_url) as db:
        alice = get_or_create_student(db, "Alice").id
        bob = get_or_create_student(db, "Bob").id

    for student_id in (alice, alice, bob):
        rec = SessionRecorder(video_path="s.mov", student_id=student_id, db_url=db_url)
        rec.add_row(make_row(1))
        rec.finalize({})

    with session_scope(db_url) as db:
        assert len(list_sessions(db, student_id=alice)) == 2
        assert len(list_sessions(db)) == 3


def test_dataframe_reprend_les_colonnes_du_csv(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_rows([make_row(1), make_row(2, ttc_s="")])
    session_id = recorder.finalize({})

    with session_scope(db_url) as db:
        df = detections_dataframe(db, session_id)

    assert list(df.columns)[:4] == ["Frame", "track_id", "Categorie", "Objet"]
    assert len(df) == 2
    assert df["d_m"].iloc[0] == pytest.approx(5.44)


def test_summary_reconstruit_le_format_json(db_url):
    recorder = SessionRecorder(video_path="s.mov", db_url=db_url)
    recorder.add_row(make_row(1))
    session_id = recorder.finalize(
        {
            "distance_m_mean": 5.4,
            "ttc_s_p50": 12.0,
            "n_events": 1,
            "ttc_histogram": {"bins": [0, 1, 2], "counts": [0, 1]},
        }
    )

    with session_scope(db_url) as db:
        summary = statistics_as_summary(get_statistics(db, session_id))

    assert summary["distance_m_mean"] == pytest.approx(5.4)
    assert summary["ttc_s_p50"] == pytest.approx(12.0)
    assert summary["ttc_histogram"]["counts"] == [0, 1]


def test_isolation_entre_seances(db_url):
    ids = []
    for n in range(2):
        rec = SessionRecorder(video_path=f"s{n}.mov", db_url=db_url)
        rec.add_rows([make_row(i) for i in range(1, 4 + n)])
        ids.append(rec.finalize({}))

    with session_scope(db_url) as db:
        counts = [
            db.query(Detection).filter(Detection.session_id == sid).count()
            for sid in ids
        ]
        assert counts == [3, 4]
        assert db.query(Session).count() == 2
