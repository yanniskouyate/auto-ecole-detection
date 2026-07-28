"""Modèle de caméra sténopé et génération de calibration cohérente.

Le fichier ``config/homography_default.json`` livré avec le projet associe
4 points image à un rectangle sol supposé de 3,5 m × 16 m. Ces correspondances
ne sont réalisables par **aucune dashcam frontale** : l'échelle latérale y
décroît d'un facteur 1,69 entre 4 m et 20 m alors qu'un sténopé impose un
facteur 5 (l'échelle varie en :math:`1/Y`). La seule pose compatible serait une
caméra inclinée à ~86° vers le sol.

Ce module construit à la place une calibration dérivée de paramètres physiques
explicites (focale, hauteur de caméra, inclinaison), donc cohérente par
construction.

Géométrie
---------
Repère monde : :math:`X` latéral (droite), :math:`Y` longitudinal (avant),
:math:`Z` vertical (haut). Caméra en :math:`(0, 0, h)`, inclinée de
:math:`\\theta` sous l'horizontale. Pour un point sol :math:`(X, Y, 0)` :

.. math::

    Z_c = Y\\cos\\theta + h\\sin\\theta, \\qquad
    Y_c = h\\cos\\theta - Y\\sin\\theta

.. math::

    u = c_x + f\\frac{X}{Z_c}, \\qquad v = c_y + f\\frac{Y_c}{Z_c}

L'horizon (:math:`Y \\to \\infty`) est en :math:`v_\\infty = c_y - f\\tan\\theta`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Caméra sténopé regardant un sol plan."""

    image_size: tuple[int, int] = (1280, 832)
    horizontal_fov_deg: float = 60.0
    height_m: float = 1.25
    pitch_deg: float = 4.0  # positif = incliné vers le sol

    @property
    def focal_px(self) -> float:
        """Focale en pixels déduite du champ horizontal."""
        w = self.image_size[0]
        return (w * 0.5) / math.tan(math.radians(self.horizontal_fov_deg) * 0.5)

    @property
    def principal_point(self) -> tuple[float, float]:
        """Centre optique (pixels)."""
        w, h = self.image_size
        return w * 0.5, h * 0.5

    @property
    def horizon_v(self) -> float:
        """Ordonnée image de la ligne d'horizon."""
        _, cy = self.principal_point
        return cy - self.focal_px * math.tan(math.radians(self.pitch_deg))

    def ground_to_pixel(self, X: float, Y: float) -> tuple[float, float]:
        """Projette un point sol (mètres) vers l'image (pixels)."""
        theta = math.radians(self.pitch_deg)
        cx, cy = self.principal_point
        f = self.focal_px
        z_c = Y * math.cos(theta) + self.height_m * math.sin(theta)
        y_c = self.height_m * math.cos(theta) - Y * math.sin(theta)
        if z_c <= 1e-9:
            raise ValueError(f"Point derrière la caméra : Y={Y}")
        return cx + f * X / z_c, cy + f * y_c / z_c

    def nearest_visible_y(self) -> float:
        """Distance sol la plus proche encore dans le champ (bas d'image)."""
        theta = math.radians(self.pitch_deg)
        _, cy = self.principal_point
        f = self.focal_px
        v_max = float(self.image_size[1])
        # Résout v(Y) = v_max pour Y
        k = (v_max - cy) / f
        num = self.height_m * (math.cos(theta) - k * math.sin(theta))
        den = k * math.cos(theta) + math.sin(theta)
        return num / den if den > 1e-9 else float("inf")

    def calibration_dict(
        self,
        near_y_m: float | None = None,
        far_y_m: float = 30.0,
        half_width_m: float = 1.75,
    ) -> dict:
        """Produit un dictionnaire de calibration au schéma du projet.

        Les 4 points image sont *calculés* par projection du rectangle sol :
        l'homographie estimée à partir d'eux reproduit exactement ce sténopé.
        """
        if near_y_m is None:
            near_y_m = max(self.nearest_visible_y() * 1.15, 3.0)

        ground = [
            (-half_width_m, near_y_m),
            (half_width_m, near_y_m),
            (half_width_m, far_y_m),
            (-half_width_m, far_y_m),
        ]
        image = [self.ground_to_pixel(x, y) for x, y in ground]

        return {
            "image_size": list(self.image_size),
            "description": (
                "Calibration dérivée d'un modèle sténopé explicite "
                f"(FOV {self.horizontal_fov_deg}°, hauteur {self.height_m} m, "
                f"inclinaison {self.pitch_deg}°). Cohérente par construction : "
                "l'échelle latérale décroît bien en 1/Y."
            ),
            "image_points": [[round(u, 3), round(v, 3)] for u, v in image],
            "ground_points": [[round(x, 3), round(y, 3)] for x, y in ground],
            "ego_position": [0.0, 0.0],
            "lane_width_m": half_width_m * 2.0,
            "camera": {
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "height_m": self.height_m,
                "pitch_deg": self.pitch_deg,
                "focal_px": round(self.focal_px, 3),
                "horizon_v_px": round(self.horizon_v, 3),
            },
            "notes": (
                "Généré par auto_ecole_math.simulation.camera. "
                "Remplacer les valeurs par une vraie mesure terrain "
                "(repères au sol de longueur connue) avant usage en production."
            ),
        }

    def save_calibration(self, path: str | Path, **kwargs) -> Path:
        """Écrit la calibration au format JSON du projet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.calibration_dict(**kwargs)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def lateral_scale_px_per_m(homography, y_m: float, half_width_m: float = 1.75) -> float:
    """Échelle latérale (px/m) mesurée sur une homographie, à la distance ``y_m``."""
    left = homography.ground_to_pixel(-half_width_m, y_m)
    right = homography.ground_to_pixel(half_width_m, y_m)
    return float(abs(right[0] - left[0]) / (2 * half_width_m))


def check_pinhole_consistency(
    homography, y_near_m: float = 5.0, y_far_m: float = 25.0, tolerance: float = 0.15
) -> dict:
    """Teste si une homographie est compatible avec une caméra frontale.

    Pour un sténopé l'échelle latérale varie en :math:`1/Y`, donc
    :math:`s(Y_1)/s(Y_2) \\simeq Y_2/Y_1`. Un écart important signale des
    correspondances de calibration erronées — et donc des distances fausses.

    Returns
    -------
    dict
        ``expected_ratio``, ``measured_ratio``, ``relative_error``,
        ``consistent`` et ``implied_far_y_m`` (distance réelle du point
        lointain si le point proche est correct).
    """
    s_near = lateral_scale_px_per_m(homography, y_near_m)
    s_far = lateral_scale_px_per_m(homography, y_far_m)
    measured = s_near / s_far if s_far > 1e-9 else float("inf")
    expected = y_far_m / y_near_m
    rel_err = abs(measured - expected) / expected
    return {
        "y_near_m": y_near_m,
        "y_far_m": y_far_m,
        "scale_near_px_per_m": s_near,
        "scale_far_px_per_m": s_far,
        "measured_ratio": measured,
        "expected_ratio": expected,
        "relative_error": rel_err,
        "consistent": bool(rel_err <= tolerance),
        "implied_far_y_m": y_near_m * measured,
    }


def default_synthetic_calibration_path() -> Path:
    """Chemin canonique de la calibration synthétique du projet."""
    root = Path(__file__).resolve().parents[3]
    return root / "config" / "homography_pinhole.json"


def ensure_synthetic_calibration(
    camera: PinholeCamera | None = None, path: str | Path | None = None
) -> Path:
    """Crée la calibration sténopé si elle n'existe pas encore, et la retourne."""
    path = Path(path) if path is not None else default_synthetic_calibration_path()
    if not path.exists():
        (camera or PinholeCamera()).save_calibration(path)
    return path
