"""Génération de séquences vidéo annotées à partir des scénarios.

Produit, pour chaque scénario :

- une vidéo ``.mp4`` de dashcam ;
- un CSV de **vérité terrain** reprenant les colonnes du rapport d'analyse
  (``X_m``, ``Y_m``, ``d_m``, ``ttc_s``…) plus la boîte englobante en pixels ;
- un JSON de métadonnées (erreur commise, frame de collision, calibration).

La vérité terrain n'est pas une estimation : les positions sont celles qui ont
servi au rendu. Comparer la sortie du pipeline à ce CSV mesure donc l'erreur
réelle de l'homographie, du filtre de Kalman et du TTC.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from auto_ecole_math.geometry.homography import HomographyEstimator
from auto_ecole_math.kinematics.ttc import ttc_from_track
from auto_ecole_math.pipeline import RuleConfig
from auto_ecole_math.simulation.camera import (
    PinholeCamera,
    ensure_synthetic_calibration,
)
from auto_ecole_math.simulation.renderer import DashcamRenderer, RenderConfig
from auto_ecole_math.simulation.scenarios import (
    Scenario,
    build_scenario,
    footprints_overlap,
)

#: Codecs tentés dans l'ordre. ``avc1`` (H.264) est lisible par les navigateurs
#: et les lecteurs courants ; ``mp4v`` sert de repli si FFMPEG ne l'expose pas.
VIDEO_CODECS = ("avc1", "mp4v")


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Ouvre un encodeur vidéo, en essayant les codecs par ordre de préférence."""
    for codec in VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Aucun codec disponible parmi {VIDEO_CODECS} pour écrire {path}"
    )


def _axis_spans(width_m: float, length_m: float, yaw_rad: float) -> tuple[float, float]:
    """Encombrement d'un rectangle orienté, projeté sur les axes X et Y."""
    c, s = abs(np.cos(yaw_rad)), abs(np.sin(yaw_rad))
    return width_m * c + length_m * s, width_m * s + length_m * c


TRUTH_HEADER = [
    "Frame",
    "t_s",
    "actor_id",
    "Objet",
    "Categorie",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "X_m",
    "Y_m",
    "vx_mps",
    "vy_mps",
    "speed_mps",
    "d_m",
    "lateral_m",
    "ttc_s",
    "error_type",
    "is_error_window",
    "frames_to_collision",
]


@dataclass
class GeneratorConfig:
    """Paramètres de génération.

    Par défaut le rendu s'appuie sur une calibration **sténopé cohérente**
    (voir :mod:`auto_ecole_math.simulation.camera`) et non sur
    ``config/homography_default.json``, dont les correspondances ne
    correspondent à aucune caméra frontale réaliste. Passer explicitement
    ``calibration_path`` permet de rendre avec la calibration du projet, par
    exemple pour comparer le pipeline à lui-même.
    """

    fps: float = 30.0
    output_dir: Path = Path("outputs/dataset")
    calibration_path: str | None = None
    camera: PinholeCamera | None = None
    render: RenderConfig | None = None
    #: Écrit aussi les labels au format YOLO (entraînement)
    export_yolo: bool = False
    draw_debug_boxes: bool = False


@dataclass
class GeneratedSequence:
    """Résultat d'une génération."""

    name: str
    video_path: Path
    truth_path: Path
    meta_path: Path
    n_frames: int
    collision_frame: int | None
    error_type: str | None


class DatasetGenerator:
    """Rend les scénarios en vidéos annotées."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.cfg = config or GeneratorConfig()
        self.camera = self.cfg.camera or PinholeCamera()

        if self.cfg.calibration_path is not None:
            calib = self.cfg.calibration_path
        else:
            calib = str(ensure_synthetic_calibration(self.camera))

        self.homography = HomographyEstimator.from_json(calib)
        self.calibration_path = calib

        render_cfg = self.cfg.render or RenderConfig(
            image_size=self.camera.image_size,
            camera_height_m=self.camera.height_m,
            min_render_y_m=max(self.camera.nearest_visible_y(), 2.0),
        )
        self.renderer = DashcamRenderer(self.homography, render_cfg)
        self.rules = RuleConfig()
        self.cfg.output_dir = Path(self.cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Vérité terrain -------------------------------------------------

    def _truth_rows(
        self, scenario: Scenario, frame_idx: int, t: float
    ) -> tuple[list[dict[str, Any]], bool]:
        """Lignes de vérité terrain d'une frame, et présence d'un contact."""
        ex, ey, evx, evy = scenario.ego.state(t)
        ego_cls = scenario.ego.as_actor().cls
        rows: list[dict[str, Any]] = []
        collided = False

        for actor in scenario.actors:
            if not actor.is_present(t):
                continue
            ax, ay, avx, avy = actor.state(t)

            # Repère sol ego-relatif : celui que le pipeline reconstruit
            rel_x, rel_y = ax - ex, ay - ey
            rel_vx, rel_vy = avx - evx, avy - evy

            # Emprise projetée sur les axes : un véhicule en travers occupe sa
            # longueur latéralement, pas sa largeur.
            yaw_world = float(np.arctan2(avx, avy)) if np.hypot(avx, avy) > 0.1 else 0.0
            span_x, span_y = _axis_spans(actor.cls.width_m, actor.cls.length_m, yaw_world)
            if footprints_overlap(
                ex, ey, ego_cls.width_m, ego_cls.length_m,
                ax, ay, span_x, span_y,
            ):
                collided = True

            ttc = ttc_from_track(
                (rel_x, rel_y),
                (rel_vx, rel_vy),
                ego_position=(0.0, 0.0),
                ego_velocity=(0.0, 0.0),
                mode="range",
            )

            # Un véhicule est orienté selon son propre déplacement (repère monde),
            # pas selon sa vitesse relative à l'ego.
            yaw = float(np.arctan2(avx, avy)) if np.hypot(avx, avy) > 0.1 else 0.0
            corners = self.renderer.actor_corners(rel_x, rel_y, actor.cls, yaw)
            if corners is None:
                continue  # hors champ : pas d'annotation
            x1, y1, x2, y2 = self.renderer.bounding_box(corners)

            in_error = scenario.error_start_s <= t <= scenario.error_end_s
            rows.append(
                {
                    "Frame": frame_idx,
                    "t_s": round(t, 4),
                    "actor_id": actor.actor_id,
                    "Objet": actor.cls.label,
                    "Categorie": self.rules.categories.get(actor.cls.label, "Autre"),
                    "x1_px": round(x1, 2),
                    "y1_px": round(y1, 2),
                    "x2_px": round(x2, 2),
                    "y2_px": round(y2, 2),
                    "X_m": round(rel_x, 5),
                    "Y_m": round(rel_y, 5),
                    "vx_mps": round(rel_vx, 5),
                    "vy_mps": round(rel_vy, 5),
                    "speed_mps": round(float(np.hypot(rel_vx, rel_vy)), 5),
                    "d_m": round(ttc["d_m"], 5),
                    "lateral_m": round(rel_x, 5),
                    "ttc_s": round(ttc["ttc_s"], 4)
                    if np.isfinite(ttc["ttc_s"])
                    else "",
                    "error_type": scenario.error_type or "",
                    "is_error_window": int(in_error and scenario.error_type is not None),
                    "frames_to_collision": "",  # complété après coup
                    "_corners": corners,
                }
            )

        return rows, collided

    # -- Rendu ----------------------------------------------------------

    def _render_frame(
        self, scenario: Scenario, t: float, rows: list[dict[str, Any]]
    ) -> np.ndarray:
        """Compose l'image d'une frame."""
        _, ey, _, evy = scenario.ego.state(t)
        canvas = self.renderer.draw_scene(ey)

        # Peintre : les objets lointains d'abord
        for row in sorted(rows, key=lambda r: -float(r["Y_m"])):
            actor = next(
                a for a in scenario.actors if a.actor_id == row["actor_id"]
            )
            self.renderer.draw_actor(canvas, row["_corners"], actor.cls)
            if self.cfg.draw_debug_boxes:
                cv2.rectangle(
                    canvas,
                    (int(row["x1_px"]), int(row["y1_px"])),
                    (int(row["x2_px"]), int(row["y2_px"])),
                    (0, 255, 255),
                    1,
                )

        speed_kmh = evy * 3.6
        hud = [f"{scenario.name}  t={t:5.2f}s  v={speed_kmh:5.1f} km/h"]
        if rows:
            nearest = min(rows, key=lambda r: float(r["d_m"]))
            ttc_txt = f"{nearest['ttc_s']}s" if nearest["ttc_s"] != "" else "inf"
            hud.append(f"d={float(nearest['d_m']):.1f} m  TTC={ttc_txt}")
        self.renderer.draw_hud(canvas, hud)
        return canvas

    # -- Génération -----------------------------------------------------

    def generate(self, scenario: Scenario, suffix: str = "") -> GeneratedSequence:
        """Rend un scénario complet et écrit vidéo + annotations."""
        fps = self.cfg.fps
        n_frames = int(round(scenario.duration_s * fps))
        stem = f"{scenario.name}{suffix}"
        out_dir = self.cfg.output_dir
        video_path = out_dir / f"{stem}.mp4"

        writer = _open_writer(video_path, fps, self.renderer.cfg.image_size)

        all_rows: list[dict[str, Any]] = []
        collision_frame: int | None = None

        try:
            for frame_idx in range(1, n_frames + 1):
                t = (frame_idx - 1) / fps
                rows, collided = self._truth_rows(scenario, frame_idx, t)
                if collided and collision_frame is None:
                    collision_frame = frame_idx

                writer.write(self._render_frame(scenario, t, rows))
                all_rows.extend(rows)

                # Après contact la scène n'a plus de sens : on arrête net
                if collision_frame is not None:
                    n_frames = frame_idx
                    break
        finally:
            writer.release()

        for row in all_rows:
            row.pop("_corners", None)
            if collision_frame is not None:
                row["frames_to_collision"] = collision_frame - row["Frame"]

        truth_path = out_dir / f"{stem}_truth.csv"
        with truth_path.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=TRUTH_HEADER)
            csv_writer.writeheader()
            csv_writer.writerows(all_rows)

        meta = {
            "scenario": scenario.name,
            "description": scenario.description,
            "error_type": scenario.error_type,
            "error_start_s": scenario.error_start_s,
            "error_end_s": scenario.error_end_s,
            "expect_collision": scenario.expect_collision,
            "collision_frame": collision_frame,
            "collision_t_s": (collision_frame - 1) / fps if collision_frame else None,
            "fps": fps,
            "n_frames": n_frames,
            "n_annotations": len(all_rows),
            "image_size": list(self.renderer.cfg.image_size),
            "camera_height_m": self.renderer.cfg.camera_height_m,
            "calibration_path": str(self.calibration_path),
            "ego_speed_mps": scenario.ego.speed_mps,
            "video": video_path.name,
            "truth_csv": truth_path.name,
        }
        meta_path = out_dir / f"{stem}_meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if self.cfg.export_yolo:
            self._export_yolo_labels(stem, all_rows)

        return GeneratedSequence(
            name=stem,
            video_path=video_path,
            truth_path=truth_path,
            meta_path=meta_path,
            n_frames=n_frames,
            collision_frame=collision_frame,
            error_type=scenario.error_type,
        )

    def _export_yolo_labels(self, stem: str, rows: list[dict[str, Any]]) -> None:
        """Écrit les annotations au format YOLO (une ligne par objet)."""
        labels_dir = self.cfg.output_dir / "labels" / stem
        labels_dir.mkdir(parents=True, exist_ok=True)
        w, h = self.renderer.cfg.image_size
        classes = sorted({r["Objet"] for r in rows})
        class_index = {name: i for i, name in enumerate(classes)}

        by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_frame.setdefault(row["Frame"], []).append(row)

        for frame_idx, frame_rows in by_frame.items():
            lines = []
            for row in frame_rows:
                x1, y1 = float(row["x1_px"]), float(row["y1_px"])
                x2, y2 = float(row["x2_px"]), float(row["y2_px"])
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
                lines.append(
                    f"{class_index[row['Objet']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
            (labels_dir / f"frame_{frame_idx:05d}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

        (labels_dir / "classes.txt").write_text(
            "\n".join(classes) + "\n", encoding="utf-8"
        )

    def generate_named(self, name: str, variation: int = 0) -> GeneratedSequence:
        """Construit puis rend un scénario du catalogue."""
        scenario = build_scenario(name, variation)
        return self.generate(scenario, suffix=f"_v{variation:03d}")
