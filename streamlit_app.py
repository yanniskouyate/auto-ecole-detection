"""Interface Streamlit — analyse de conduite et affichage du rapport.

Usage (depuis la racine du dépôt, avec le venv du projet)::

    source .venv/bin/activate
    PYTHONPATH=src python -m streamlit run streamlit_app.py

Important : ne pas lancer ``streamlit`` depuis Anaconda (souvent sans OpenCV).
Utiliser ``python -m streamlit`` après activation de ``.venv``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_VIDEO = ROOT / "auto_ecole_test.mov"
DEFAULT_CSV = ROOT / "rapport_conduite.csv"
DEFAULT_STATS = ROOT / "rapport_stats.json"
DEFAULT_MODEL = ROOT / "yolo11n.pt"
DEFAULT_CALIB = ROOT / "config" / "homography_default.json"


def _probe_cv2() -> tuple[bool, str | None]:
    """Teste la disponibilité d'OpenCV sans faire planter l'UI."""
    try:
        import cv2  # noqa: F401

        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _probe_db() -> tuple[bool, str | None]:
    """Teste la disponibilité de la couche base de données."""
    try:
        from auto_ecole_math.database.db import init_db

        init_db()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


st.set_page_config(
    page_title="Auto-École Vision — Rapport",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analyse de conduite — Rapport quantitatif")
st.caption(
    "YOLO → homographie (mètres) → Kalman → TTC / freinage → statistiques de séance"
)

_DB_OK, _DB_ERR = _probe_db()
_CV2_OK, _CV2_ERR = _probe_cv2()
if not _CV2_OK:
    st.error(
        "**OpenCV (cv2) introuvable** pour cet interpréteur Python.\n\n"
        f"`{sys.executable}`\n\n"
        "Relance avec le venv du projet :\n\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "PYTHONPATH=src python -m streamlit run streamlit_app.py\n"
        "```\n\n"
        f"Détail : `{_CV2_ERR}`"
    )


def _load_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, suffix: str = "", digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.{digits}f}{suffix}"


def render_stats(stats: dict) -> None:
    """Affiche les indicateurs clés de la séance."""
    if not stats:
        st.info("Aucune statistique disponible.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Événements", int(stats.get("n_events", 0)))
    c2.metric("Distance moyenne", _fmt(stats.get("distance_m_mean"), " m"))
    c3.metric("Écart-type distance", _fmt(stats.get("distance_m_std"), " m"))
    c4.metric("TTC médian", _fmt(stats.get("ttc_s_p50"), " s"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Distance min", _fmt(stats.get("distance_m_min"), " m"))
    c6.metric("Latéral moyen", _fmt(stats.get("lateral_m_mean"), " m"))
    c7.metric("Vitesse relative moy.", _fmt(stats.get("speed_mps_mean"), " m/s"))
    c8.metric("Confiance lissée moy.", _fmt(stats.get("confidence_mean"), "", 3))

    hist = stats.get("ttc_histogram") or {}
    bins = hist.get("bins") or []
    counts = hist.get("counts") or []
    if bins and counts and len(bins) == len(counts) + 1:
        labels = [f"{bins[i]}–{bins[i+1]}s" for i in range(len(counts))]
        hist_df = pd.DataFrame({"Intervalle TTC": labels, "Occurrences": counts})
        st.subheader("Distribution des Time-To-Collision")
        st.bar_chart(hist_df.set_index("Intervalle TTC"))


def render_report_table(df: pd.DataFrame) -> None:
    """Tableau filtrable + graphiques temporels."""
    if df.empty:
        st.warning("Le rapport CSV est vide.")
        return

    st.subheader("Événements détectés")
    categories = sorted(df["Categorie"].dropna().unique().tolist()) if "Categorie" in df else []
    objects = sorted(df["Objet"].dropna().unique().tolist()) if "Objet" in df else []

    f1, f2 = st.columns(2)
    selected_cat = f1.multiselect("Filtrer par catégorie", categories, default=categories)
    selected_obj = f2.multiselect("Filtrer par objet", objects, default=objects)

    view = df.copy()
    if selected_cat and "Categorie" in view.columns:
        view = view[view["Categorie"].isin(selected_cat)]
    if selected_obj and "Objet" in view.columns:
        view = view[view["Objet"].isin(selected_obj)]

    st.dataframe(view, use_container_width=True, height=360)

    chart_cols = [c for c in ["d_m", "ttc_s", "lateral_m", "speed_mps"] if c in view.columns]
    if "Frame" in view.columns and chart_cols:
        st.subheader("Évolution temporelle")
        plot_df = view[["Frame", *chart_cols]].copy()
        for col in chart_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=chart_cols, how="all").set_index("Frame")
        if not plot_df.empty:
            st.line_chart(plot_df)

    mask = pd.Series(False, index=view.index)
    if "margin_m" in view.columns:
        margin = pd.to_numeric(view["margin_m"], errors="coerce")
        mask = mask | (margin < 0)
    if "ttc_s" in view.columns:
        ttc = pd.to_numeric(view["ttc_s"], errors="coerce")
        mask = mask | (ttc < 3)
    risky = view[mask]
    st.subheader("Alertes sécurité (TTC < 3 s ou marge de freinage < 0)")
    if risky.empty:
        st.success("Aucune alerte critique sur les filtres actuels.")
    else:
        st.dataframe(risky, use_container_width=True, height=240)


def render_history_table(rows: list[dict]) -> None:
    """Progression d'un élève au fil de ses séances."""
    if not rows:
        return
    st.subheader("Progression de l'élève")
    hist = pd.DataFrame(rows)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    plot_cols = [c for c in ("distance_mean", "ttc_p50") if c in hist.columns]
    numeric = hist.dropna(subset=plot_cols, how="all")
    if plot_cols and len(numeric) > 1:
        st.line_chart(numeric.set_index("session_id")[plot_cols])


def resolve_student_id(name: str | None) -> int | None:
    """Retrouve (ou crée) l'élève saisi dans la barre latérale."""
    if not name or not _DB_OK:
        return None
    from auto_ecole_math.database.db import session_scope
    from auto_ecole_math.database.queries import get_or_create_student

    with session_scope() as db:
        return get_or_create_student(db, name.strip()).id


def run_analysis(
    video_path: Path,
    progress_bar,
    status_text,
    *,
    student_id: int | None = None,
    label: str | None = None,
) -> tuple[Path, Path, int | None]:
    """Lance le pipeline et retourne les chemins CSV / stats + l'ID de séance."""
    import cv2
    from auto_ecole_math.pipeline import DrivingAnalysisPipeline, PipelineConfig

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "rapport_conduite.csv"
    stats_path = out_dir / "rapport_stats.json"

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    cfg = PipelineConfig(
        video_path=str(video_path),
        model_path=str(DEFAULT_MODEL),
        calibration_path=str(DEFAULT_CALIB),
        csv_path=str(csv_path),
        stats_path=str(stats_path),
        show_display=False,
        db_enabled=_DB_OK,
        student_id=student_id,
        session_label=label,
    )
    pipeline = DrivingAnalysisPipeline(cfg)

    def on_frame(frame_idx: int, _annotated, _rows) -> None:
        if n_frames > 0:
            progress_bar.progress(min(frame_idx / n_frames, 1.0))
        status_text.text(f"Analyse en cours… frame {frame_idx}/{n_frames or '?'}")

    pipeline.run(on_frame=on_frame)
    progress_bar.progress(1.0)
    status_text.text("Analyse terminée.")
    return csv_path, stats_path, pipeline.session_id


# --- Sidebar ---
with st.sidebar:
    st.header("Source")
    st.caption(f"Python : `{sys.executable}`")
    mode = st.radio(
        "Mode",
        [
            "Analyser une vidéo",
            "Historique des séances",
            "Afficher un rapport existant",
        ],
        index=0,
    )
    use_default = st.checkbox(
        "Utiliser la vidéo de test du projet",
        value=DEFAULT_VIDEO.exists(),
        disabled=not DEFAULT_VIDEO.exists(),
    )

    st.divider()
    st.header("Séance")
    if _DB_OK:
        st.caption("Enregistrement en base activé.")
        student_name = st.text_input("Élève", placeholder="Prénom Nom")
        session_label = st.text_input("Libellé", placeholder="Séance du 12 mars")
    else:
        student_name = None
        session_label = None
        st.warning(f"Base indisponible — mode CSV seul.\n\n`{_DB_ERR}`")

# --- Main ---
csv_path: Path | None = None
stats_path: Path | None = None
db_df: pd.DataFrame | None = None
db_summary: dict | None = None

if mode == "Historique des séances":
    if not _DB_OK:
        st.error(f"Base de données indisponible : `{_DB_ERR}`")
    else:
        from auto_ecole_math.database.db import session_scope
        from auto_ecole_math.database.queries import (
            detections_dataframe,
            get_statistics,
            list_sessions,
            statistics_as_summary,
            student_progress,
        )

        with session_scope() as db:
            sessions = list_sessions(db)
            options = {
                f"#{s.id} — {s.label or Path(s.video_path).name}"
                f" — {s.student.name if s.student else 'élève inconnu'}"
                f" — {s.date:%Y-%m-%d %H:%M}": s.id
                for s in sessions
            }

        if not options:
            st.info(
                "Aucune séance en base. Lance une analyse, ou importe un rapport "
                "existant avec `python scripts/migrate_csv_to_db.py`."
            )
        else:
            chosen_label = st.selectbox("Séance", list(options))
            chosen_id = options[chosen_label]
            with session_scope() as db:
                session_row = next(s for s in list_sessions(db) if s.id == chosen_id)
                db_summary = statistics_as_summary(get_statistics(db, chosen_id))
                db_df = detections_dataframe(db, chosen_id)
                progression = (
                    student_progress(db, session_row.student_id)
                    if session_row.student_id
                    else []
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Frames analysées", session_row.n_frames or 0)
            m2.metric("Durée", _fmt(session_row.duration_s, " s", 1))
            m3.metric("Modèle", session_row.model_version or "—")
            render_history_table(progression)

elif mode == "Afficher un rapport existant":
    candidates = []
    for p in (ROOT / "outputs" / "rapport_conduite.csv", DEFAULT_CSV):
        if p.exists():
            candidates.append(p)
    if not candidates:
        st.warning(
            "Aucun rapport trouvé. Lance d'abord une analyse "
            "(`PYTHONPATH=src python analyse_video.py --no-display`) "
            "ou utilise le mode « Analyser une vidéo »."
        )
    else:
        chosen = st.selectbox("Rapport CSV", candidates, format_func=lambda p: str(p))
        csv_path = Path(chosen)
        stats_candidate = csv_path.with_name("rapport_stats.json")
        if not stats_candidate.exists() and DEFAULT_STATS.exists():
            stats_candidate = DEFAULT_STATS
        stats_path = stats_candidate if stats_candidate.exists() else None
        if DEFAULT_VIDEO.exists():
            st.video(str(DEFAULT_VIDEO))

else:
    uploaded = None
    if not use_default:
        uploaded = st.file_uploader("Vidéo dashcam", type=["mp4", "mov", "avi"])

    video_to_run: Path | None = None

    if use_default and DEFAULT_VIDEO.exists():
        video_to_run = DEFAULT_VIDEO
        st.video(str(DEFAULT_VIDEO))
    elif uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded.read())
        tmp_file.close()
        video_to_run = Path(tmp_file.name)
        st.video(uploaded)

    if video_to_run is not None:
        if st.button("Lancer l'analyse IA", type="primary", disabled=not _CV2_OK):
            progress = st.progress(0.0)
            status = st.empty()
            with st.spinner("Chargement du modèle et analyse…"):
                try:
                    csv_path, stats_path, session_id = run_analysis(
                        video_to_run,
                        progress,
                        status,
                        student_id=resolve_student_id(student_name),
                        label=session_label or None,
                    )
                    st.session_state["last_csv"] = str(csv_path)
                    st.session_state["last_stats"] = str(stats_path)
                    if session_id is not None:
                        st.success(
                            f"Rapport généré et enregistré en base (séance #{session_id})."
                        )
                    else:
                        st.success("Rapport généré (CSV uniquement).")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Échec de l'analyse : {exc}")
                    csv_path, stats_path = None, None

    if "last_csv" in st.session_state and csv_path is None:
        csv_path = Path(st.session_state["last_csv"])
        stats_path = Path(st.session_state.get("last_stats", ""))

# --- Affichage rapport ---
if db_df is not None:
    st.divider()
    st.header("Rapport de conduite")
    render_stats(db_summary or {})
    render_report_table(db_df)
    st.download_button(
        "Télécharger le CSV de cette séance",
        data=db_df.to_csv(index=False).encode("utf-8"),
        file_name="rapport_conduite.csv",
        mime="text/csv",
    )

elif csv_path is not None and csv_path.exists():
    st.divider()
    st.header("Rapport de conduite")

    stats = _load_stats(stats_path) if stats_path and stats_path.exists() else {}
    render_stats(stats)

    df = pd.read_csv(csv_path)
    render_report_table(df)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Télécharger le CSV",
            data=csv_path.read_bytes(),
            file_name="rapport_conduite.csv",
            mime="text/csv",
        )
    with d2:
        if stats_path and stats_path.exists():
            st.download_button(
                "Télécharger les stats JSON",
                data=stats_path.read_bytes(),
                file_name="rapport_stats.json",
                mime="application/json",
            )
elif mode == "Analyser une vidéo":
    st.info("Choisis une vidéo puis clique sur **Lancer l'analyse IA** pour générer le rapport.")
