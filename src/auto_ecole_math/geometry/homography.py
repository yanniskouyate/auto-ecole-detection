"""Géométrie projective : homographie image ↔ plan de la route.

Le passage pixels → mètres repose sur l'hypothèse d'un **plan sol plat**.
Une homographie :math:`H \\in \\mathbb{R}^{3\\times 3}` relie un point image
homogène :math:`\\mathbf{x} = (u, v, 1)^\\top` à un point sol
:math:`\\mathbf{X} = (X, Y, 1)^\\top` via

.. math::

    \\mathbf{X} \\sim H \\mathbf{x},
    \\qquad
    X = \\frac{(H\\mathbf{x})_1}{(H\\mathbf{x})_3},\\;
    Y = \\frac{(H\\mathbf{x})_2}{(H\\mathbf{x})_3}.

L'estimation de :math:`H` utilise la DLT (Direct Linear Transform) avec
RANSAC lorsque :math:`n \\ge 4` correspondances sont disponibles
(:func:`cv2.findHomography`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .calibration import (
    HomographyCalibration,
    default_calibration_path,
    load_calibration,
)


def bbox_bottom_center(xyxy: Sequence[float]) -> np.ndarray:
    """Point de contact approximatif objet–sol : milieu du bord bas de la bbox.

    .. math::

        (u_b, v_b) = \\Bigl(\\frac{x_1 + x_2}{2},\\, y_2\\Bigr)

    Parameters
    ----------
    xyxy :
        Boîte ``[x1, y1, x2, y2]`` en pixels.
    """
    x1, _, x2, y2 = map(float, xyxy)
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float64)


class HomographyEstimator:
    """Estime et applique une homographie plan-sol (IPM métrique).

    Parameters
    ----------
    calibration :
        Correspondances image ↔ sol. Si ``None``, charge
        ``config/homography_default.json``.
    method :
        Méthode OpenCV (``cv2.RANSAC`` par défaut).
    ransac_reproj_threshold :
        Seuil de reprojection RANSAC en pixels.
    """

    def __init__(
        self,
        calibration: HomographyCalibration | None = None,
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        if calibration is None:
            calibration = load_calibration(default_calibration_path())
        self.calibration = calibration
        self.ego_position = np.asarray(calibration.ego_position, dtype=np.float64).reshape(2)

        self.H, self.mask = cv2.findHomography(
            calibration.image_points.astype(np.float64),
            calibration.ground_points.astype(np.float64),
            method=method,
            ransacReprojThreshold=ransac_reproj_threshold,
        )
        if self.H is None:
            raise RuntimeError("Homography estimation failed (cv2.findHomography returned None)")

        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> HomographyEstimator:
        """Construit l'estimateur depuis un fichier de calibration JSON."""
        return cls(calibration=load_calibration(path), **kwargs)

    def pixel_to_ground(self, u: float, v: float) -> np.ndarray:
        """Projette un pixel :math:`(u, v)` vers le plan sol :math:`(X, Y)` en mètres.

        .. math::

            \\mathbf{X} = H \\mathbf{x},\\quad
            (X, Y) = \\Bigl(\\frac{X_1}{X_3}, \\frac{X_2}{X_3}\\Bigr)
        """
        return self.pixels_to_ground(np.array([[u, v]], dtype=np.float64))[0]

    def pixels_to_ground(self, pixels: np.ndarray) -> np.ndarray:
        """Projette un ensemble de pixels shape ``(N, 2)`` vers le sol ``(N, 2)``."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def ground_to_pixel(self, X: float, Y: float) -> np.ndarray:
        """Projette un point sol :math:`(X, Y)` vers l'image :math:`(u, v)`."""
        return self.grounds_to_pixels(np.array([[X, Y]], dtype=np.float64))[0]

    def grounds_to_pixels(self, grounds: np.ndarray) -> np.ndarray:
        """Projette des points sol shape ``(N, 2)`` vers l'image ``(N, 2)``."""
        pts = np.asarray(grounds, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_inv)
        return out.reshape(-1, 2)

    @staticmethod
    def ground_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
        """Distance euclidienne sur le plan sol (mètres).

        .. math::

            d = \\| (X_1, Y_1) - (X_2, Y_2) \\|_2
        """
        a = np.asarray(p1, dtype=np.float64).reshape(2)
        b = np.asarray(p2, dtype=np.float64).reshape(2)
        return float(np.linalg.norm(a - b))

    def distance_from_ego(self, ground_xy: Sequence[float]) -> float:
        """Distance ego → objet sur le plan sol (mètres)."""
        return self.ground_distance(self.ego_position, ground_xy)

    def longitudinal_distance(self, ground_xy: Sequence[float]) -> float:
        """Distance longitudinale :math:`|Y - Y_{\\mathrm{ego}}|` (mètres)."""
        y = float(np.asarray(ground_xy, dtype=np.float64).reshape(2)[1])
        return abs(y - float(self.ego_position[1]))

    def lateral_offset(self, ground_xy: Sequence[float]) -> float:
        """Écart latéral :math:`X - X_{\\mathrm{ego}}` (mètres, signé)."""
        x = float(np.asarray(ground_xy, dtype=np.float64).reshape(2)[0])
        return x - float(self.ego_position[0])

    def bbox_to_ground(self, xyxy: Sequence[float]) -> np.ndarray:
        """Projette le pied de bbox (bottom-center) vers le plan sol."""
        uv = bbox_bottom_center(xyxy)
        return self.pixel_to_ground(float(uv[0]), float(uv[1]))

    def reprojection_errors(self) -> np.ndarray:
        """Erreurs de reprojection (mètres) sur les points de calibration.

        .. math::

            e_i = \\| H \\mathbf{x}_i - \\mathbf{X}_i \\|_2
        """
        pred = self.pixels_to_ground(self.calibration.image_points)
        return np.linalg.norm(pred - self.calibration.ground_points, axis=1)

    def mean_reprojection_error(self) -> float:
        """Erreur moyenne de reprojection en mètres."""
        return float(np.mean(self.reprojection_errors()))
