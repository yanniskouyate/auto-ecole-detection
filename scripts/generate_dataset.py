"""Génère un jeu de séquences de conduite annotées.

Chaque séquence produit une vidéo ``.mp4``, un CSV de vérité terrain et un JSON
de métadonnées. Aucun simulateur externe n'est requis : le rendu s'appuie sur
OpenCV et sur le modèle de caméra du projet.

Usage::

    PYTHONPATH=src python scripts/generate_dataset.py
    PYTHONPATH=src python scripts/generate_dataset.py --variations 10
    PYTHONPATH=src python scripts/generate_dataset.py \\
        --scenarios pieton_non_anticipe refus_priorite --yolo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.simulation.camera import PinholeCamera  # noqa: E402
from auto_ecole_math.simulation.generator import (  # noqa: E402
    DatasetGenerator,
    GeneratorConfig,
)
from auto_ecole_math.simulation.scenarios import SCENARIO_BUILDERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Génération de séquences annotées")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help=f"Sous-ensemble à générer (défaut : tous). Choix : {sorted(SCENARIO_BUILDERS)}",
    )
    parser.add_argument("--variations", type=int, default=1, help="Variantes par scénario")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "dataset"))
    parser.add_argument(
        "--calibration",
        default=None,
        help="Calibration à utiliser (défaut : sténopé cohérent auto-généré)",
    )
    parser.add_argument("--fov", type=float, default=60.0, help="Champ horizontal (degrés)")
    parser.add_argument("--camera-height", type=float, default=1.25)
    parser.add_argument("--pitch", type=float, default=4.0, help="Inclinaison caméra (degrés)")
    parser.add_argument("--yolo", action="store_true", help="Exporte aussi les labels YOLO")
    parser.add_argument(
        "--debug-boxes", action="store_true", help="Superpose les boîtes englobantes"
    )
    args = parser.parse_args()

    names = args.scenarios or list(SCENARIO_BUILDERS)
    unknown = [n for n in names if n not in SCENARIO_BUILDERS]
    if unknown:
        raise SystemExit(
            f"Scénario(s) inconnu(s) : {unknown}. Disponibles : {sorted(SCENARIO_BUILDERS)}"
        )

    camera = PinholeCamera(
        horizontal_fov_deg=args.fov,
        height_m=args.camera_height,
        pitch_deg=args.pitch,
    )
    generator = DatasetGenerator(
        GeneratorConfig(
            fps=args.fps,
            output_dir=Path(args.out),
            calibration_path=args.calibration,
            camera=camera,
            export_yolo=args.yolo,
            draw_debug_boxes=args.debug_boxes,
        )
    )

    print(f"Calibration : {Path(generator.calibration_path).name}")
    print(f"Sortie      : {generator.cfg.output_dir}")
    print()
    print(f"{'séquence':34s} {'frames':>7s} {'collision':>10s} {'annot.':>8s} {'taille':>9s}")
    print("-" * 72)

    index = []
    for name in names:
        for variation in range(args.variations):
            seq = generator.generate_named(name, variation)
            meta = json.loads(seq.meta_path.read_text(encoding="utf-8"))
            size_kb = seq.video_path.stat().st_size // 1024
            print(
                f"{seq.name:34s} {seq.n_frames:7d} "
                f"{str(seq.collision_frame or '—'):>10s} "
                f"{meta['n_annotations']:8d} {size_kb:6d} Ko"
            )
            index.append(
                {
                    "name": seq.name,
                    "scenario": name,
                    "video": seq.video_path.name,
                    "truth_csv": seq.truth_path.name,
                    "meta": seq.meta_path.name,
                    "error_type": seq.error_type,
                    "collision_frame": seq.collision_frame,
                    "n_frames": seq.n_frames,
                    "n_annotations": meta["n_annotations"],
                }
            )

    index_path = generator.cfg.output_dir / "dataset_index.json"
    index_path.write_text(
        json.dumps(
            {
                "calibration": str(generator.calibration_path),
                "fps": args.fps,
                "camera": {
                    "horizontal_fov_deg": camera.horizontal_fov_deg,
                    "height_m": camera.height_m,
                    "pitch_deg": camera.pitch_deg,
                },
                "sequences": index,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    total_annotations = sum(s["n_annotations"] for s in index)
    n_collisions = sum(1 for s in index if s["collision_frame"])
    print("-" * 72)
    print(
        f"{len(index)} séquences, {total_annotations} annotations, "
        f"{n_collisions} avec collision"
    )
    print(f"Index : {index_path}")
    print()
    print("Mesurer l'erreur du pipeline sur ce jeu :")
    print(f"  PYTHONPATH=src python scripts/validate_geometry.py --dir {args.out}")


if __name__ == "__main__":
    main()
