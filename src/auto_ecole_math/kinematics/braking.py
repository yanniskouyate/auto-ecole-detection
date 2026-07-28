"""Distance de freinage théorique (cinématique élémentaire).

Sous décélération constante :math:`a > 0` et temps de réaction :math:`t_r` :

.. math::

    d_{\\mathrm{frein}}(v) = v\\, t_r + \\frac{v^2}{2 a}

où :math:`v` est la vitesse relative d'approche (m/s).
Valeurs typiques : :math:`t_r \\approx 1.0\\,\\mathrm{s}`,
:math:`a \\in [6, 8]\\,\\mathrm{m/s^2}` (freinage d'urgence sec).
"""

from __future__ import annotations


def braking_distance(
    speed_mps: float,
    reaction_time_s: float = 1.0,
    deceleration_mps2: float = 7.0,
) -> float:
    """Distance de freinage théorique (mètres).

    .. math::

        d_{\\mathrm{frein}} = v t_r + \\frac{v^2}{2 a}

    Parameters
    ----------
    speed_mps :
        Vitesse :math:`v \\ge 0` (m/s). On utilise :math:`|v|`.
    reaction_time_s :
        Temps de réaction :math:`t_r` (s).
    deceleration_mps2 :
        Décélération constante :math:`a > 0` (m/s²).
    """
    v = abs(float(speed_mps))
    tr = float(reaction_time_s)
    a = float(deceleration_mps2)
    if a <= 0:
        raise ValueError("deceleration_mps2 must be > 0")
    return v * tr + (v * v) / (2.0 * a)


def stopping_margin(
    distance_m: float,
    speed_mps: float,
    reaction_time_s: float = 1.0,
    deceleration_mps2: float = 7.0,
) -> dict[str, float]:
    """Marge de sécurité : distance disponible moins distance de freinage.

    .. math::

        m = d - d_{\\mathrm{frein}}(v)

    Returns
    -------
    dict
        ``d_frein_m``, ``margin_m``, ``is_safe`` (1.0 si :math:`m > 0`).
    """
    d_brake = braking_distance(speed_mps, reaction_time_s, deceleration_mps2)
    margin = float(distance_m) - d_brake
    return {
        "d_frein_m": d_brake,
        "margin_m": margin,
        "is_safe": 1.0 if margin > 0 else 0.0,
    }


def two_second_rule_distance(speed_mps: float, gap_s: float = 2.0) -> float:
    """Distance minimale recommandée (règle des :math:`N` secondes).

    .. math::

        d_{2s} = v \\cdot \\Delta t
    """
    return abs(float(speed_mps)) * float(gap_s)
