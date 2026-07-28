"""Mesure l'erreur du pipeline géométrique contre la vérité terrain simulée.

Les vidéos générées viennent avec les positions exactes qui ont servi au rendu.
On peut donc quantifier ce que le pipeline reconstruit réellement — sans passer
par YOLO, dont les erreurs de détection masqueraient celles de la géométrie.

Chaîne testée : boîte englobante → homographie → Kalman → TTC.

Usage::

    PYTHONPATH=src python scripts/validate_geometry.py
    PYTHONPATH=src python scripts/validate_geometry.py \\
        --truth outputs/dataset/conduite_saine_v000_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.geometry.homography import HomographyEstimator  # noqa: E402
from auto_ecole_math.kinematics.ttc import ttc_from_track  # noqa: E402
from auto_ecole_math.simulation.actors import ACTOR_CLASSES  # noqa: E402
from auto_ecole_math.tracking.multi_tracker import (  # noqa: E402
    Detection,
    MultiObjectTracker,
)

#: Au-delà de ce TTC la grandeur n'a plus de portée pratique (objet non menaçant)
TTC_RELEVANT_S = 10.0

#: Nombre de mises à jour avant que la vitesse de Kalman soit exploitable
WARMUP_FRAMES = 15


def load_truth(path: Path) -> tuple[list[dict], dict]:
    """Charge le CSV de vérité terrain et ses métadonnées associées."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    meta_path = path.with_name(path.name.replace("_truth.csv", "_meta.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return rows, meta


def evaluate(truth_path: Path) -> dict:
    """Rejoue une séquence et compare chaque grandeur à la vérité."""
    rows, meta = load_truth(truth_path)
    if not rows:
        return {
            "sequence": truth_path.stem.replace("_truth", ""),
            "scenario": meta.get("scenario"),
            "n_annotations": 0,
            "metrics": {},
            "biais_y_par_classe": {},
        }

    homography = HomographyEstimator.from_json(meta["calibration_path"])
    fps = float(meta.get("fps", 30.0))
    tracker = MultiObjectTracker()
    tracker.dt = 1.0 / fps

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["Frame"])].append(row)

    errors = {
        "x_m": [], "y_m": [], "d_m": [], "ttc_s": [], "ttc_s_warmup": [],
        "y_m_corrected": [], "speed_mps": [],
    }
    per_class: dict[str, list[float]] = defaultdict(list)
    track_age: dict[int, int] = defaultdict(int)

    for frame_idx in sorted(by_frame):
        frame_rows = by_frame[frame_idx]
        detections = [
            Detection(
                xy=homography.bbox_to_ground(
                    [float(r["x1_px"]), float(r["y1_px"]),
                     float(r["x2_px"]), float(r["y2_px"])]
                ),
                label=r["Objet"],
                confidence=1.0,
                xyxy=np.array(
                    [float(r["x1_px"]), float(r["y1_px"]),
                     float(r["x2_px"]), float(r["y2_px"])],
                    dtype=np.float64,
                ),
                frame=frame_idx,
            )
            for r in frame_rows
        ]
        tracks = tracker.update(detections, dt=1.0 / fps)
        for track in tracks:
            if track.time_since_update == 0:
                track_age[track.track_id] += 1

        for row in frame_rows:
            true_xy = np.array([float(row["X_m"]), float(row["Y_m"])])
            # Apparie la piste la plus proche de la position vraie
            live = [t for t in tracks if t.time_since_update == 0]
            if not live:
                continue
            track = min(live, key=lambda t: np.linalg.norm(t.position - true_xy))

            est_x, est_y = float(track.position[0]), float(track.position[1])
            errors["x_m"].append(est_x - true_xy[0])
            errors["y_m"].append(est_y - true_xy[1])

            # La bbox touche le sol au bord AVANT de l'objet, pas en son centre
            half_len = ACTOR_CLASSES[row["Objet"]].length_m * 0.5
            errors["y_m_corrected"].append((est_y + half_len) - true_xy[1])
            per_class[row["Objet"]].append(est_y - true_xy[1])

            ttc = ttc_from_track(
                track.position, track.velocity,
                ego_position=(0.0, 0.0), ego_velocity=(0.0, 0.0), mode="range",
            )
            errors["d_m"].append(ttc["d_m"] - float(row["d_m"]))
            errors["speed_mps"].append(track.speed - float(row["speed_mps"]))
            # Le TTC n'est comparé que dans la plage utile à la sécurité : au-delà
            # il tend vers l'infini et une différence de vitesse infime y produit
            # un écart de plusieurs milliers de secondes, sans portée pratique.
            if row["ttc_s"] not in ("", None) and np.isfinite(ttc["ttc_s"]):
                true_ttc = float(row["ttc_s"])
                if 0.0 < true_ttc <= TTC_RELEVANT_S:
                    delta = ttc["ttc_s"] - true_ttc
                    mature = track_age[track.track_id] >= WARMUP_FRAMES
                    errors["ttc_s" if mature else "ttc_s_warmup"].append(delta)

    def stats(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "biais": float(np.mean(arr)),
            "rms": float(np.sqrt(np.mean(arr**2))),
            "abs_max": float(np.max(np.abs(arr))),
        }

    return {
        "sequence": truth_path.stem.replace("_truth", ""),
        "scenario": meta.get("scenario"),
        "n_annotations": len(rows),
        "metrics": {k: stats(v) for k, v in errors.items()},
        "biais_y_par_classe": {
            label: round(float(np.mean(v)), 3) for label, v in per_class.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erreur du pipeline géométrique face à la vérité terrain"
    )
    parser.add_argument("--truth", default=None, help="Un CSV _truth.csv précis")
    parser.add_argument("--dir", default="outputs/dataset", help="Dossier du dataset")
    args = parser.parse_args()

    if args.truth:
        paths = [Path(args.truth)]
    else:
        paths = sorted(Path(args.dir).glob("*_truth.csv"))
    if not paths:
        raise SystemExit(
            f"Aucun fichier _truth.csv dans {args.dir}. "
            "Génère d'abord un dataset : python scripts/generate_dataset.py"
        )

    print(f"{'séquence':32s} {'n':>6s} {'biais Y':>9s} {'RMS Y':>8s} "
          f"{'RMS Y corr':>11s} {'RMS X':>8s} {'TTC établi':>11s} {'TTC amorce':>11s}")
    print("-" * 100)

    aggregate: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        report = evaluate(path)
        m = report["metrics"]
        if not m:
            print(f"{report['sequence']:32s} {'—':>6s}   (aucune annotation : objet hors champ)")
            continue
        print(
            f"{report['sequence']:32s} {report['n_annotations']:6d} "
            f"{m['y_m'].get('biais', float('nan')):9.3f} "
            f"{m['y_m'].get('rms', float('nan')):8.3f} "
            f"{m['y_m_corrected'].get('rms', float('nan')):11.3f} "
            f"{m['x_m'].get('rms', float('nan')):8.3f} "
            f"{m['ttc_s'].get('rms', float('nan')):11.3f} "
            f"{m['ttc_s_warmup'].get('rms', float('nan')):11.3f}"
        )
        for key in ("x_m", "y_m", "y_m_corrected", "ttc_s"):
            if m[key].get("n"):
                aggregate[key].append(m[key]["rms"])

    print("-" * 100)
    print(f"Unités : mètres, sauf TTC en secondes (comparé seulement si TTC vrai ≤ {TTC_RELEVANT_S:.0f} s).")
    print()
    print("Biais Y : vaut la demi-longueur de l'objet. La boîte englobante touche le")
    print("  sol à son bord avant, pas en son centre — le pipeline sous-estime donc")
    print("  systématiquement la distance de ~2,25 m pour une voiture.")
    print("« RMS Y corr » : erreur restante une fois ce biais retiré, soit l'erreur")
    print("  propre à l'homographie et au filtre.")
    print(f"« TTC amorce » : pistes de moins de {WARMUP_FRAMES} frames, où la vitesse de")
    print("  Kalman n'a pas convergé. Le TTC y est inexploitable — à ne pas remonter")
    print("  comme alerte tant que la piste n'est pas établie.")


if __name__ == "__main__":
    main()
