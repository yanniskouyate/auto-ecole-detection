"""Rendu d'une image de dashcam à partir de la géométrie sol.

Le rendu est l'**inverse exact** de la chaîne d'analyse : on part de positions
connues en mètres et on les projette vers l'image avec :math:`H^{-1}`, la même
homographie que celle utilisée par :mod:`auto_ecole_math.geometry`. Un objet
posé à :math:`(X, Y)` sera donc reprojeté par le pipeline en :math:`(X, Y)` à
l'erreur numérique près : c'est ce qui rend la vérité terrain exploitable pour
mesurer l'erreur de l'homographie, du Kalman et du TTC.

Hauteurs
--------
Une homographie plan-sol ne décrit que le sol ; elle ne suffit pas à projeter
un point en hauteur. On utilise la relation pinhole classique : pour une
caméra à la hauteur :math:`h_c` au-dessus d'un sol plat, un objet de hauteur
:math:`h` posé au sol vérifie

.. math::

    v_{top} = v_\\infty + (v_{bottom} - v_\\infty)\\left(1 - \\frac{h}{h_c}\\right)

où :math:`v_\\infty` est la ligne d'horizon (image de la droite à l'infini du
plan sol). Un objet aussi haut que la caméra a bien son sommet sur l'horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from auto_ecole_math.geometry.homography import HomographyEstimator
from auto_ecole_math.simulation.actors import ActorClass

#: Couleurs BGR de la scène
SKY_TOP = (150, 110, 70)
SKY_BOTTOM = (215, 190, 165)
GROUND_SIDE = (95, 115, 90)
ROAD = (68, 68, 72)
MARKING = (225, 230, 235)


@dataclass
class RenderConfig:
    """Paramètres géométriques et visuels du rendu."""

    image_size: tuple[int, int] = (1280, 832)
    camera_height_m: float = 1.25
    lane_width_m: float = 3.5
    #: Géométrie sol non rendue en deçà (le capot masque le très proche)
    min_render_y_m: float = 2.0
    #: Portée du rendu de la chaussée
    max_render_y_m: float = 220.0
    #: Marquage axial : longueur peinte / période (norme ~3 m / 13 m)
    dash_length_m: float = 3.0
    dash_period_m: float = 13.0
    draw_hud: bool = True


class DashcamRenderer:
    """Dessine une image de dashcam depuis des positions sol en mètres."""

    def __init__(
        self,
        homography: HomographyEstimator,
        config: RenderConfig | None = None,
    ) -> None:
        self.h = homography
        self.cfg = config or RenderConfig()
        self.width, self.height = self.cfg.image_size
        self._horizon_line = self._compute_horizon_line()
        self._sky = self._build_sky()

    # -- Géométrie ------------------------------------------------------

    def _compute_horizon_line(self) -> np.ndarray:
        """Droite d'horizon en coordonnées homogènes image.

        Les points à l'infini du plan sol, de directions :math:`(1,0,0)` et
        :math:`(0,1,0)`, s'envoient sur les deux premières colonnes de
        :math:`H^{-1}`. Leur produit vectoriel donne la droite qui les joint.
        """
        p1 = self.h.H_inv[:, 0]
        p2 = self.h.H_inv[:, 1]
        line = np.cross(p1, p2)
        norm = np.linalg.norm(line[:2])
        return line / norm if norm > 1e-12 else line

    def horizon_v(self, u: float) -> float:
        """Ordonnée image de l'horizon à la colonne ``u``."""
        a, b, c = self._horizon_line
        if abs(b) < 1e-12:
            return 0.0
        return float(-(a * u + c) / b)

    def ground_to_image(self, points_m: np.ndarray) -> np.ndarray:
        """Projette des points sol ``(N, 2)`` en pixels ``(N, 2)``."""
        return self.h.grounds_to_pixels(np.asarray(points_m, dtype=np.float64))

    def raise_point(self, uv: np.ndarray, height_m: float) -> np.ndarray:
        """Élève un point image posé au sol de ``height_m`` mètres."""
        u, v = float(uv[0]), float(uv[1])
        v_inf = self.horizon_v(u)
        ratio = 1.0 - height_m / max(self.cfg.camera_height_m, 1e-6)
        return np.array([u, v_inf + (v - v_inf) * ratio], dtype=np.float64)

    # -- Décor ----------------------------------------------------------

    def _build_sky(self) -> np.ndarray:
        """Fond dégradé ciel → sol, calculé une fois pour toutes."""
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        v_mid = self.horizon_v(self.width * 0.5)
        for v in range(self.height):
            if v < v_mid:
                t = v / max(v_mid, 1.0)
                color = [
                    int(SKY_TOP[c] + (SKY_BOTTOM[c] - SKY_TOP[c]) * t) for c in range(3)
                ]
            else:
                color = list(GROUND_SIDE)
            canvas[v, :] = color
        return canvas

    def _road_polygon(self, y_ego_m: float) -> np.ndarray | None:
        """Quadrilatère de la chaussée, en pixels."""
        half = self.cfg.lane_width_m * 1.5  # 2 voies + accotement
        y0 = self.cfg.min_render_y_m
        y1 = self.cfg.max_render_y_m
        corners = np.array(
            [[-half, y0], [half, y0], [half, y1], [-half, y1]], dtype=np.float64
        )
        return _valid_polygon(self.ground_to_image(corners))

    def _draw_dashes(
        self, canvas: np.ndarray, x_m: float, y_ego_m: float, width_m: float = 0.15
    ) -> None:
        """Marquage discontinu défilant avec l'avancée de l'ego."""
        period = self.cfg.dash_period_m
        # Décalage de phase : les traits défilent vers l'observateur
        phase = y_ego_m % period
        y = self.cfg.min_render_y_m - phase
        while y < self.cfg.max_render_y_m:
            y_start = max(y, self.cfg.min_render_y_m)
            y_end = min(y + self.cfg.dash_length_m, self.cfg.max_render_y_m)
            if y_end > y_start:
                quad = np.array(
                    [
                        [x_m - width_m, y_start],
                        [x_m + width_m, y_start],
                        [x_m + width_m, y_end],
                        [x_m - width_m, y_end],
                    ],
                    dtype=np.float64,
                )
                poly = _valid_polygon(self.ground_to_image(quad))
                if poly is not None:
                    cv2.fillPoly(canvas, [poly], MARKING, lineType=cv2.LINE_AA)
            y += period

    def _draw_solid_line(
        self, canvas: np.ndarray, x_m: float, width_m: float = 0.15
    ) -> None:
        """Ligne continue (bord de chaussée)."""
        quad = np.array(
            [
                [x_m - width_m, self.cfg.min_render_y_m],
                [x_m + width_m, self.cfg.min_render_y_m],
                [x_m + width_m, self.cfg.max_render_y_m],
                [x_m - width_m, self.cfg.max_render_y_m],
            ],
            dtype=np.float64,
        )
        poly = _valid_polygon(self.ground_to_image(quad))
        if poly is not None:
            cv2.fillPoly(canvas, [poly], MARKING, lineType=cv2.LINE_AA)

    def draw_scene(self, y_ego_m: float) -> np.ndarray:
        """Fond complet : ciel, chaussée, marquages."""
        canvas = self._sky.copy()
        road = self._road_polygon(y_ego_m)
        if road is not None:
            cv2.fillPoly(canvas, [road], ROAD, lineType=cv2.LINE_AA)

        half_lane = self.cfg.lane_width_m * 0.5
        self._draw_solid_line(canvas, -half_lane)  # bord gauche
        self._draw_dashes(canvas, half_lane, y_ego_m)  # séparateur de voies
        self._draw_solid_line(canvas, half_lane + self.cfg.lane_width_m)  # bord droit
        return canvas

    # -- Acteurs --------------------------------------------------------

    def actor_corners(
        self, x_m: float, y_m: float, cls: ActorClass, yaw_rad: float = 0.0
    ) -> np.ndarray | None:
        """Les 8 sommets image du pavé englobant, ou ``None`` si hors champ.

        Ordre : 4 sommets au sol puis les 4 sommets supérieurs correspondants.
        ``yaw_rad`` est l'orientation du véhicule, mesurée depuis l'axe
        longitudinal : un véhicule traversant l'intersection présente son flanc
        et non son arrière.
        """
        if y_m < self.cfg.min_render_y_m or y_m > self.cfg.max_render_y_m:
            return None

        hw, hl = cls.width_m * 0.5, cls.length_m * 0.5
        local = np.array(
            [[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=np.float64
        )
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        rotation = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64)
        base = local @ rotation.T + np.array([x_m, y_m], dtype=np.float64)

        if np.any(base[:, 1] < self.cfg.min_render_y_m * 0.5):
            return None

        bottom = self.ground_to_image(base)
        if not np.all(np.isfinite(bottom)):
            return None
        top = np.array([self.raise_point(p, cls.height_m) for p in bottom])
        corners = np.vstack([bottom, top])
        return corners if np.all(np.isfinite(corners)) else None

    def bounding_box(self, corners: np.ndarray) -> tuple[float, float, float, float]:
        """Boîte englobante alignée ``(x1, y1, x2, y2)`` — format YOLO/COCO."""
        x1 = float(np.min(corners[:, 0]))
        y1 = float(np.min(corners[:, 1]))
        x2 = float(np.max(corners[:, 0]))
        y2 = float(np.max(corners[:, 1]))
        return x1, y1, x2, y2

    def draw_actor(self, canvas: np.ndarray, corners: np.ndarray, cls: ActorClass) -> None:
        """Dessine un acteur en pavé ombré."""
        bottom, top = corners[:4], corners[4:]

        silhouette = _valid_polygon(
            cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
        )
        if silhouette is None:
            return
        cv2.fillPoly(canvas, [silhouette], cls.color, lineType=cv2.LINE_AA)

        # Face arrière (arête la plus proche de la caméra) éclaircie
        rear = _valid_polygon(
            np.vstack([bottom[0], bottom[1], top[1], top[0]])
        )
        if rear is not None:
            lighter = tuple(min(255, int(c * 1.25)) for c in cls.color)
            cv2.fillPoly(canvas, [rear], lighter, lineType=cv2.LINE_AA)

        if cls.label == "person":
            head = _valid_polygon(np.vstack([top[0], top[1], top[2], top[3]]))
            if head is not None:
                cv2.fillPoly(canvas, [head], (70, 70, 150), lineType=cv2.LINE_AA)
        elif cls.label in ("car", "truck", "bus"):
            # Bandeau de vitre sur le haut de la face arrière
            band_bottom = bottom[:2] + (top[:2] - bottom[:2]) * 0.55
            band = _valid_polygon(
                np.vstack([band_bottom[0], band_bottom[1], top[1], top[0]])
            )
            if band is not None:
                cv2.fillPoly(canvas, [band], (45, 45, 50), lineType=cv2.LINE_AA)

        outline = _valid_polygon(
            cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
        )
        if outline is not None:
            cv2.polylines(
                canvas, [outline], True, (25, 25, 25), 1, lineType=cv2.LINE_AA
            )

    def draw_hud(self, canvas: np.ndarray, lines: list[str]) -> None:
        """Bandeau d'information en haut à gauche."""
        if not self.cfg.draw_hud:
            return
        for i, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (16, 30 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


def _valid_polygon(points: np.ndarray) -> np.ndarray | None:
    """Convertit en polygone entier dessinable, ou ``None`` si dégénéré.

    Les projections proches de l'horizon peuvent produire des coordonnées
    immenses : on les rejette plutôt que de laisser OpenCV déborder.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return None
    if np.any(np.abs(pts) > 1e5):
        return None
    return np.round(pts).astype(np.int32)
