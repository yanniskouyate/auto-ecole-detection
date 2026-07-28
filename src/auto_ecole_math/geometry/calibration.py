"""Chargement et sauvegarde de calibrations d'homographie plan-sol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HomographyCalibration:
    """Correspondances image ↔ sol pour estimer une homographie.

    Attributes
    ----------
    image_points :
        Points image :math:`(u_i, v_i)` en pixels, shape ``(N, 2)``, :math:`N \\ge 4`.
    ground_points :
        Points sol :math:`(X_i, Y_i)` en mètres, shape ``(N, 2)``.
    image_size :
        ``(width, height)`` optionnel de la vidéo source.
    ego_position :
        Position de l'ego dans le plan sol, typiquement :math:`(0, 0)`.
    metadata :
        Champs libres (description, notes, etc.).
    """

    image_points: np.ndarray
    ground_points: np.ndarray
    image_size: tuple[int, int] | None = None
    ego_position: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        img = np.asarray(self.image_points, dtype=np.float64)
        grd = np.asarray(self.ground_points, dtype=np.float64)
        if img.ndim != 2 or img.shape[1] != 2:
            raise ValueError("image_points must have shape (N, 2)")
        if grd.shape != img.shape:
            raise ValueError("ground_points must match image_points shape")
        if img.shape[0] < 4:
            raise ValueError("at least 4 point correspondences are required")
        object.__setattr__(self, "image_points", img)
        object.__setattr__(self, "ground_points", grd)
        if self.ego_position is None:
            object.__setattr__(self, "ego_position", np.zeros(2, dtype=np.float64))
        else:
            object.__setattr__(
                self, "ego_position", np.asarray(self.ego_position, dtype=np.float64).reshape(2)
            )


def load_calibration(path: str | Path) -> HomographyCalibration:
    """Charge une calibration JSON.

    Format attendu::

        {
          "image_points": [[u, v], ...],
          "ground_points": [[X, Y], ...],
          "image_size": [width, height],
          "ego_position": [0.0, 0.0]
        }
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    image_size = data.get("image_size")
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))

    metadata = {
        k: v
        for k, v in data.items()
        if k not in {"image_points", "ground_points", "image_size", "ego_position"}
    }

    return HomographyCalibration(
        image_points=np.asarray(data["image_points"], dtype=np.float64),
        ground_points=np.asarray(data["ground_points"], dtype=np.float64),
        image_size=image_size,
        ego_position=np.asarray(data.get("ego_position", [0.0, 0.0]), dtype=np.float64),
        metadata=metadata or None,
    )


def save_calibration(calibration: HomographyCalibration, path: str | Path) -> None:
    """Sauvegarde une calibration au format JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "image_points": calibration.image_points.tolist(),
        "ground_points": calibration.ground_points.tolist(),
        "ego_position": (
            calibration.ego_position.tolist()
            if calibration.ego_position is not None
            else [0.0, 0.0]
        ),
    }
    if calibration.image_size is not None:
        payload["image_size"] = list(calibration.image_size)
    if calibration.metadata:
        payload.update(calibration.metadata)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def default_calibration_path() -> Path:
    """Chemin du fichier de calibration par défaut du dépôt."""
    return Path(__file__).resolve().parents[3] / "config" / "homography_default.json"
