"""Time-To-Collision (TTC) à partir de la dynamique relative.

Pour une distance scalaire :math:`d(t)` (typiquement longitudinale) :

.. math::

    \\mathrm{TTC}(t) =
    \\begin{cases}
    \\dfrac{d(t)}{-\\dot d(t)} & \\text{si } \\dot d(t) < 0,\\\\
    +\\infty & \\text{sinon.}
    \\end{cases}

La dérivée :math:`\\dot d` peut provenir de l'état Kalman
:math:`(\\dot X, \\dot Y)` projeté sur la direction ego→cible, ou d'une
différence finie filtrée.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def time_to_collision(distance: float, closing_rate: float) -> float:
    """Calcule le TTC à partir de :math:`d` et :math:`\\dot d`.

    Parameters
    ----------
    distance :
        Distance :math:`d \\ge 0` (m).
    closing_rate :
        Dérivée :math:`\\dot d` (m/s). Négatif = rapprochement.

    Returns
    -------
    float
        TTC en secondes, ou ``numpy.inf`` si pas de collision.
    """
    d = float(distance)
    dd = float(closing_rate)
    if d < 0:
        d = abs(d)
    if dd >= 0:
        return float("inf")
    return d / (-dd)


def relative_closing_rate(
    position: Sequence[float],
    velocity: Sequence[float],
    ego_position: Sequence[float] = (0.0, 0.0),
    ego_velocity: Sequence[float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Distance et taux de fermeture relatifs ego–cible.

    Soit :math:`\\mathbf{r} = \\mathbf{p}_{\\mathrm{obj}} - \\mathbf{p}_{\\mathrm{ego}}`
    et :math:`\\mathbf{v}_{\\mathrm{rel}} = \\mathbf{v}_{\\mathrm{obj}} - \\mathbf{v}_{\\mathrm{ego}}`.

    .. math::

        d = \\|\\mathbf{r}\\|_2,\\qquad
        \\dot d = \\frac{\\mathbf{r}\\cdot \\mathbf{v}_{\\mathrm{rel}}}{d}

    Returns
    -------
    distance, closing_rate :
        :math:`(d, \\dot d)` en (m, m/s).
    """
    r = np.asarray(position, dtype=np.float64).reshape(2) - np.asarray(
        ego_position, dtype=np.float64
    ).reshape(2)
    v_rel = np.asarray(velocity, dtype=np.float64).reshape(2) - np.asarray(
        ego_velocity, dtype=np.float64
    ).reshape(2)
    d = float(np.linalg.norm(r))
    if d < 1e-9:
        return 0.0, 0.0
    dd = float(np.dot(r, v_rel) / d)
    return d, dd


def longitudinal_ttc(
    Y: float,
    vy: float,
    ego_Y: float = 0.0,
    ego_vy: float = 0.0,
) -> tuple[float, float, float]:
    """TTC longitudinal 1D le long de l'axe :math:`Y` (devant le véhicule).

    .. math::

        d = Y - Y_{\\mathrm{ego}},\\quad
        \\dot d = \\dot Y - \\dot Y_{\\mathrm{ego}}

    Returns
    -------
    d, dd, ttc
    """
    d = float(Y) - float(ego_Y)
    dd = float(vy) - float(ego_vy)
    # Pour un objet devant (d > 0), rapprochement si dd < 0
    if d <= 0:
        # objet derrière ou au niveau
        ttc = float("inf") if dd >= 0 else time_to_collision(abs(d), dd)
    else:
        ttc = time_to_collision(d, dd)
    return d, dd, ttc


def ttc_from_track(
    position: Sequence[float],
    velocity: Sequence[float],
    ego_position: Sequence[float] = (0.0, 0.0),
    ego_velocity: Sequence[float] = (0.0, 0.0),
    mode: str = "range",
) -> dict[str, float]:
    """TTC pour une piste suivie.

    Parameters
    ----------
    mode :
        ``"range"`` : TTC radial 2D ; ``"longitudinal"`` : axe :math:`Y` seul.
    """
    if mode == "longitudinal":
        d, dd, ttc = longitudinal_ttc(
            float(position[1]),
            float(velocity[1]),
            ego_Y=float(ego_position[1]),
            ego_vy=float(ego_velocity[1]),
        )
    else:
        d, dd = relative_closing_rate(position, velocity, ego_position, ego_velocity)
        ttc = time_to_collision(d, dd)
    return {"d_m": d, "d_dot_mps": dd, "ttc_s": ttc}
