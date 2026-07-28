#!/usr/bin/env python3
"""Démo Phase 1 : projection YOLO → plan sol (mètres) via homographie.

Usage (depuis la racine du dépôt)::

    PYTHONPATH=src python scripts/demo_homography.py
    PYTHONPATH=src python scripts/demo_homography.py --no-display --max-frames 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.geometry.homography import HomographyEstimator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Démo homographie YOLO → mètres")
    parser.add_argument("--video", default=str(ROOT / "auto_ecole_test.mov"))
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument(
        "--calib",
        default=str(ROOT / "config" / "homography_default.json"),
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = toute la vidéo")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    est = HomographyEstimator.from_json(args.calib)
    print(f"Erreur reprojection moyenne : {est.mean_reprojection_error():.4f} m")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir {args.video}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy()
            ground = est.bbox_to_ground(xyxy)
            d = est.distance_from_ego(ground)
            label = results[0].names[int(box.cls[0])]
            text = f"{label} d={d:.1f}m ({ground[0]:.1f},{ground[1]:.1f})"
            x1, y1 = int(xyxy[0]), int(xyxy[1])
            cv2.putText(
                annotated,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
            # Draw bottom-center contact point
            uv = ((xyxy[0] + xyxy[2]) * 0.5, xyxy[3])
            cv2.circle(annotated, (int(uv[0]), int(uv[1])), 5, (0, 255, 255), -1)

        if frame_idx % 15 == 0:
            print(f"Frame {frame_idx}: overlay distances métriques OK")

        if not args.no_display:
            cv2.imshow("Demo Homographie", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("Démo homographie terminée.")


if __name__ == "__main__":
    main()
