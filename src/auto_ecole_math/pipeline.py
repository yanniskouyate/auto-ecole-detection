"""Pipeline d'analyse mathématique : détection → géométrie → tracking → cinématique.

Orchestre YOLOv11, l'homographie plan-sol, le filtre de Kalman multi-objets,
le TTC / freinage et l'agrégation statistique d'une séance de conduite.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# Ensure src/ is importable when running from repo root
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auto_ecole_math.geometry.calibration import default_calibration_path
from auto_ecole_math.geometry.homography import HomographyEstimator
from auto_ecole_math.kinematics.braking import stopping_margin
from auto_ecole_math.kinematics.ttc import ttc_from_track
from auto_ecole_math.tracking.multi_tracker import Detection, MultiObjectTracker
from auto_ecole_math.uncertainty.confidence import TrackConfidenceFilter
from auto_ecole_math.uncertainty.session_stats import SessionStatsCollector


@dataclass
class RuleConfig:
    """Seuils métier auto-école (classes COCO ciblées)."""

    class_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "stop sign": 0.80,
            "bicycle": 0.60,
            "person": 0.70,
            "car": 0.50,
            "truck": 0.50,
            "bus": 0.50,
            "motorcycle": 0.55,
        }
    )
    categories: dict[str, str] = field(
        default_factory=lambda: {
            "stop sign": "Signalisation",
            "bicycle": "Usager Vulnérable",
            "person": "Usager Vulnérable",
            "car": "Véhicule",
            "truck": "Véhicule",
            "bus": "Véhicule",
            "motorcycle": "Usager Vulnérable",
        }
    )


@dataclass
class PipelineConfig:
    """Configuration globale du pipeline."""

    video_path: str = "auto_ecole_test.mov"
    model_path: str = "yolo11n.pt"
    calibration_path: str | None = None
    csv_path: str = "rapport_conduite.csv"
    stats_path: str = "rapport_stats.json"
    show_display: bool = True
    reaction_time_s: float = 1.0
    deceleration_mps2: float = 7.0
    ego_speed_mps: float = 0.0  # sans odométrie : relatif pur (ego fixe dans le plan)
    rules: RuleConfig = field(default_factory=RuleConfig)

    # --- Persistance base de données ---
    db_enabled: bool = True
    db_url: str | None = None  # None → SQLite local (voir database.db)
    student_id: int | None = None
    instructor_id: int | None = None
    session_label: str | None = None


CSV_HEADER = [
    "Frame",
    "track_id",
    "Categorie",
    "Objet",
    "Confiance",
    "Confiance_lissee",
    "Position_X_px",
    "X_m",
    "Y_m",
    "vx_mps",
    "vy_mps",
    "speed_mps",
    "d_m",
    "lateral_m",
    "ttc_s",
    "d_frein_m",
    "margin_m",
]


class DrivingAnalysisPipeline:
    """Pipeline Master : YOLO + Homographie + Kalman + TTC + Stats."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        calib = self.config.calibration_path or str(default_calibration_path())
        self.homography = HomographyEstimator.from_json(calib)
        self.tracker = MultiObjectTracker()
        self.conf_filter = TrackConfidenceFilter()
        self.stats = SessionStatsCollector()
        self.rules = self.config.rules
        self._model = None
        # First-seen frame per track for reaction-time proxy
        self._first_seen: dict[int, int] = {}
        self._alert_frame: dict[int, int] = {}
        #: Identifiant de la séance créée en base par le dernier ``run()``
        self.session_id: int | None = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)
        return self._model

    def _filter_detections(
        self, results, frame_idx: int
    ) -> list[Detection]:
        dets: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                conf = float(box.conf[0])
                thr = self.rules.class_thresholds.get(label)
                if thr is None or conf < thr:
                    continue
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float64)
                ground = self.homography.bbox_to_ground(xyxy)
                dets.append(
                    Detection(
                        xy=ground,
                        label=label,
                        confidence=conf,
                        xyxy=xyxy,
                        frame=frame_idx,
                    )
                )
        return dets

    def _row_for_track(self, frame_idx: int, track, fps: float) -> dict[str, Any]:
        conf_info = self.conf_filter.update(track.track_id, track.confidence)
        ttc_info = ttc_from_track(
            track.position,
            track.velocity,
            ego_position=self.homography.ego_position,
            ego_velocity=(0.0, self.config.ego_speed_mps),
            mode="range",
        )
        # Relative approach speed for braking: max(0, -d_dot)
        approach = max(0.0, -ttc_info["d_dot_mps"])
        # If relative model is near-zero, fall back to track speed
        v_for_brake = approach if approach > 0.05 else track.speed
        brake = stopping_margin(
            ttc_info["d_m"],
            v_for_brake,
            reaction_time_s=self.config.reaction_time_s,
            deceleration_mps2=self.config.deceleration_mps2,
        )
        lateral = self.homography.lateral_offset(track.position)

        if track.track_id not in self._first_seen:
            self._first_seen[track.track_id] = frame_idx
        # Proxy: frames from first detection to first low-TTC event
        reaction_proxy = None
        if ttc_info["ttc_s"] < 3.0 and track.track_id not in self._alert_frame:
            self._alert_frame[track.track_id] = frame_idx
            reaction_proxy = (frame_idx - self._first_seen[track.track_id]) / max(fps, 1e-6)

        self.stats.add(
            distance_m=ttc_info["d_m"],
            lateral_m=lateral,
            ttc_s=ttc_info["ttc_s"],
            speed_mps=track.speed,
            confidence=conf_info["smoothed_conf"],
            reaction_proxy_s=reaction_proxy,
            event={
                "frame": frame_idx,
                "track_id": track.track_id,
                "label": track.label,
                "ttc_s": ttc_info["ttc_s"],
            },
        )

        x_px = float(track.last_xyxy[0]) if track.last_xyxy is not None else float("nan")
        return {
            "Frame": frame_idx,
            "track_id": track.track_id,
            "Categorie": self.rules.categories.get(track.label, "Autre"),
            "Objet": track.label,
            "Confiance": track.confidence,
            "Confiance_lissee": conf_info["smoothed_conf"],
            "Position_X_px": x_px,
            "X_m": float(track.position[0]),
            "Y_m": float(track.position[1]),
            "vx_mps": float(track.velocity[0]),
            "vy_mps": float(track.velocity[1]),
            "speed_mps": track.speed,
            "d_m": ttc_info["d_m"],
            "lateral_m": lateral,
            "ttc_s": ttc_info["ttc_s"] if np.isfinite(ttc_info["ttc_s"]) else "",
            "d_frein_m": brake["d_frein_m"],
            "margin_m": brake["margin_m"],
        }

    def _make_recorder(self, fps: float):
        """Ouvre un enregistreur base de données, ou ``None`` si désactivé.

        Un échec de connexion ne doit jamais faire perdre une analyse : on
        prévient et on continue en mode CSV seul.
        """
        if not self.config.db_enabled:
            return None
        try:
            from auto_ecole_math.database.recorder import SessionRecorder

            return SessionRecorder(
                video_path=self.config.video_path,
                student_id=self.config.student_id,
                instructor_id=self.config.instructor_id,
                calibration_path=self.config.calibration_path
                or str(default_calibration_path()),
                model_version=Path(self.config.model_path).stem,
                label=self.config.session_label,
                fps=fps,
                reaction_time_s=self.config.reaction_time_s,
                deceleration_mps2=self.config.deceleration_mps2,
                ego_speed_mps=self.config.ego_speed_mps,
                db_url=self.config.db_url,
            )
        except Exception as exc:  # noqa: BLE001 - la persistance est optionnelle
            print(f"[warn] persistance base désactivée ({type(exc).__name__}: {exc})")
            return None

    def run(
        self,
        on_frame: Callable[[int, np.ndarray, list[dict[str, Any]]], None] | None = None,
    ) -> SessionStatsCollector:
        """Exécute l'analyse complète et écrit CSV + stats JSON + base.

        Parameters
        ----------
        on_frame :
            Callback optionnel ``(frame_idx, annotated_bgr, rows)``.
        """
        model = self._load_model()
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.tracker.dt = 1.0 / fps if fps > 1e-6 else 1.0 / 30.0

        csv_path = Path(self.config.csv_path)
        rows_written = 0
        recorder = self._make_recorder(fps)
        self.session_id = None
        frame_idx = 0

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    results = model(frame, verbose=False)
                    dets = self._filter_detections(results, frame_idx)
                    tracks = self.tracker.update(dets, dt=1.0 / fps)

                    frame_rows: list[dict[str, Any]] = []
                    for track in tracks:
                        if track.time_since_update > 0:
                            continue  # log only updated tracks this frame
                        row = self._row_for_track(frame_idx, track, fps)
                        writer.writerow(row)
                        if recorder is not None:
                            recorder.add_row(row)
                        frame_rows.append(row)
                        rows_written += 1

                    annotated = results[0].plot()
                    # Overlay metric annotations
                    for row in frame_rows:
                        if row["ttc_s"] == "":
                            ttc_txt = "inf"
                        else:
                            ttc_txt = f"{float(row['ttc_s']):.1f}s"
                        label = (
                            f"ID{row['track_id']} d={row['d_m']:.1f}m "
                            f"TTC={ttc_txt}"
                        )
                        track_xyxy = _find_xyxy(tracks, row["track_id"])
                        if track_xyxy is not None:
                            x1, y1 = int(track_xyxy[0]), int(track_xyxy[1])
                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                    if on_frame is not None:
                        on_frame(frame_idx, annotated, frame_rows)

                    if self.config.show_display:
                        cv2.imshow("Analyse Mathématique Auto-École", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except BaseException:
            # Une analyse interrompue ne doit pas laisser de séance partielle
            if recorder is not None:
                recorder.abort()
            raise
        finally:
            cap.release()
            if self.config.show_display:
                cv2.destroyAllWindows()

        self.stats.save_json(self.config.stats_path)

        if recorder is not None:
            self.session_id = recorder.finalize(
                self.stats.summarize(),
                n_frames=frame_idx,
                duration_s=frame_idx / fps if fps > 1e-6 else None,
            )

        db_txt = f" ; séance #{self.session_id} en base" if self.session_id else ""
        print(
            f"Analyse terminée : {rows_written} lignes → {csv_path} ; "
            f"stats → {self.config.stats_path}{db_txt}"
        )
        return self.stats


def _find_xyxy(tracks, track_id: int):
    for t in tracks:
        if t.track_id == track_id and t.last_xyxy is not None:
            return t.last_xyxy
    return None


def run_default(
    video_path: str = "auto_ecole_test.mov",
    show_display: bool = True,
    csv_path: str = "rapport_conduite.csv",
    stats_path: str = "rapport_stats.json",
    db_enabled: bool = True,
    student_id: int | None = None,
) -> SessionStatsCollector:
    """Point d'entrée pratique pour les scripts CLI."""
    cfg = PipelineConfig(
        video_path=video_path,
        show_display=show_display,
        csv_path=csv_path,
        stats_path=stats_path,
        db_enabled=db_enabled,
        student_id=student_id,
    )
    return DrivingAnalysisPipeline(cfg).run()
