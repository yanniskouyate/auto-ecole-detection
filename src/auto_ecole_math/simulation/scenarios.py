"""Catalogue de scénarios de conduite, avec l'erreur pédagogique annotée.

Chaque scénario décrit une situation type rencontrée en leçon : l'erreur est
nommée et datée, et la collision (quand elle survient) est calculée par
recouvrement des emprises au sol — pas devinée.

Les scénarios acceptent un indice de variation qui applique une perturbation
déterministe (vitesses, distances, instants de réaction). Générer 50 variantes
d'un même scénario donne 50 séquences différentes mais reproductibles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from auto_ecole_math.simulation.actors import (
    ACTOR_CLASSES,
    Actor,
    Ego,
    LaneChange,
    Phase,
)

#: Demi-largeur de la voie de l'ego (m)
HALF_LANE = 1.75
#: Centre de la voie adjacente (m)
ADJACENT_LANE_X = 3.5


@dataclass
class Scenario:
    """Une situation de conduite complète, prête à être rendue."""

    name: str
    description: str
    duration_s: float
    ego: Ego
    actors: list[Actor] = field(default_factory=list)
    #: Étiquette d'erreur, ``None`` pour une conduite saine
    error_type: str | None = None
    #: Fenêtre pendant laquelle l'erreur est commise
    error_start_s: float = 0.0
    error_end_s: float = 0.0
    #: Le scénario est-il censé aboutir à un contact ?
    expect_collision: bool = False


def _jitter(rng: random.Random, value: float, spread: float) -> float:
    """Perturbation uniforme relative autour de ``value``."""
    return value * (1.0 + rng.uniform(-spread, spread))


# --- Scénarios -------------------------------------------------------------


def safe_following(rng: random.Random) -> Scenario:
    """Référence saine : distance de sécurité correcte, aucune erreur."""
    speed = _jitter(rng, 13.9, 0.10)
    gap = _jitter(rng, 32.0, 0.15)
    return Scenario(
        name="conduite_saine",
        description="Suivi à distance correcte, aucune erreur commise",
        duration_s=12.0,
        ego=Ego(speed_mps=speed),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["car"],
                x0_m=0.0,
                y0_m=gap,
                vy0_mps=speed,
            )
        ],
        error_type=None,
        expect_collision=False,
    )


def insufficient_following_distance(rng: random.Random) -> Scenario:
    """Le véhicule suivi freine ; l'ego était trop près et réagit trop tard."""
    speed = _jitter(rng, 16.7, 0.08)
    gap = _jitter(rng, 12.0, 0.15)
    brake_at = _jitter(rng, 3.0, 0.10)
    reaction_s = _jitter(rng, 1.2, 0.20)

    return Scenario(
        name="distance_securite",
        description="Freinage du véhicule suivi, intervalle trop court pour s'arrêter",
        duration_s=9.0,
        ego=Ego(
            speed_mps=speed,
            phases=[Phase(brake_at + reaction_s), Phase(4.0, ay_mps2=-6.5)],
        ),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["car"],
                x0_m=0.0,
                y0_m=gap,
                vy0_mps=speed,
                phases=[Phase(brake_at), Phase(3.0, ay_mps2=-7.0)],
            )
        ],
        error_type="distance_securite_insuffisante",
        error_start_s=0.0,
        error_end_s=brake_at + reaction_s,
        expect_collision=True,
    )


def pedestrian_not_anticipated(rng: random.Random) -> Scenario:
    """Un piéton traverse ; l'ego ne freine qu'au dernier moment."""
    speed = _jitter(rng, 11.0, 0.08)
    appear_at = _jitter(rng, 0.5, 0.30)
    walk_speed = _jitter(rng, 1.4, 0.12)
    start_x = _jitter(rng, 5.5, 0.10)
    # Position telle que le piéton atteigne l'axe quand l'ego y parvient
    cross_time = appear_at + start_x / walk_speed
    y_cross = speed * cross_time
    brake_at = cross_time - _jitter(rng, 0.35, 0.25)

    return Scenario(
        name="pieton_non_anticipe",
        description="Piéton traversant, freinage déclenché beaucoup trop tard",
        duration_s=max(7.0, cross_time + 2.5),
        ego=Ego(
            speed_mps=speed,
            phases=[Phase(brake_at), Phase(3.0, ay_mps2=-7.5)],
        ),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["person"],
                x0_m=start_x,
                y0_m=y_cross,
                vx0_mps=-walk_speed,
                t_appear_s=appear_at,
            )
        ],
        error_type="pieton_non_anticipe",
        error_start_s=appear_at,
        error_end_s=brake_at,
        expect_collision=True,
    )


def unchecked_lane_change(rng: random.Random) -> Scenario:
    """Rabattement sans contrôle : un véhicule plus lent occupe la voie visée.

    Le véhicule est placé *devant* l'ego, dans la voie adjacente. Un véhicule
    réellement dans l'angle mort serait hors du champ d'une caméra frontale et
    ne produirait aucune annotation exploitable : le cas observable est celui
    du rabattement sur un véhicule plus lent déjà engagé.
    """
    speed = _jitter(rng, 16.7, 0.08)
    target_speed = _jitter(rng, 14.5, 0.08)
    change_at = _jitter(rng, 3.0, 0.15)
    change_span = _jitter(rng, 2.0, 0.15)
    lead = _jitter(rng, 9.0, 0.15)

    return Scenario(
        name="rabattement_non_controle",
        description="Rabattement sur la voie adjacente occupée par un véhicule plus lent",
        duration_s=8.0,
        ego=Ego(
            speed_mps=speed,
            lane_change=LaneChange(change_at, change_at + change_span, ADJACENT_LANE_X),
        ),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["car"],
                x0_m=ADJACENT_LANE_X,
                y0_m=lead,
                vy0_mps=target_speed,
            )
        ],
        error_type="rabattement_non_controle",
        error_start_s=change_at,
        error_end_s=change_at + change_span,
        expect_collision=True,
    )


def cyclist_close_overtake(rng: random.Random) -> Scenario:
    """Dépassement de cycliste sans l'écart latéral réglementaire (1 m ville)."""
    speed = _jitter(rng, 13.9, 0.08)
    cyclist_speed = _jitter(rng, 5.0, 0.15)
    lateral = _jitter(rng, 1.45, 0.08)  # trop proche : ~0,3 m entre carrosseries
    gap = _jitter(rng, 34.0, 0.15)

    return Scenario(
        name="ecart_lateral_cycliste",
        description="Dépassement de cycliste à moins d'un mètre d'écart",
        duration_s=9.0,
        ego=Ego(speed_mps=speed),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["bicycle"],
                x0_m=lateral,
                y0_m=gap,
                vy0_mps=cyclist_speed,
            )
        ],
        error_type="ecart_lateral_insuffisant",
        error_start_s=0.0,
        error_end_s=9.0,
        expect_collision=False,
    )


def intersection_right_of_way(rng: random.Random) -> Scenario:
    """Véhicule transversal à une intersection, ego ne ralentit pas."""
    speed = _jitter(rng, 13.9, 0.08)
    cross_speed = _jitter(rng, 9.0, 0.12)
    start_x = _jitter(rng, -38.0, 0.08)
    # Instant où le véhicule transversal coupe l'axe de l'ego
    cross_time = abs(start_x) / cross_speed
    y_cross = speed * cross_time

    return Scenario(
        name="refus_priorite",
        description="Véhicule transversal à l'intersection, aucun ralentissement",
        duration_s=max(7.0, cross_time + 2.5),
        ego=Ego(speed_mps=speed),
        actors=[
            Actor(
                actor_id=1,
                cls=ACTOR_CLASSES["car"],
                x0_m=start_x,
                y0_m=y_cross,
                vx0_mps=cross_speed,
            )
        ],
        error_type="refus_priorite_intersection",
        error_start_s=0.0,
        error_end_s=cross_time,
        expect_collision=True,
    )


#: Registre des scénarios disponibles
SCENARIO_BUILDERS = {
    "conduite_saine": safe_following,
    "distance_securite": insufficient_following_distance,
    "pieton_non_anticipe": pedestrian_not_anticipated,
    "rabattement_non_controle": unchecked_lane_change,
    "ecart_lateral_cycliste": cyclist_close_overtake,
    "refus_priorite": intersection_right_of_way,
}

#: Étiquettes d'erreur, dans un ordre stable (indices de classe pour YOLO)
ERROR_LABELS = [
    "distance_securite_insuffisante",
    "pieton_non_anticipe",
    "rabattement_non_controle",
    "ecart_lateral_insuffisant",
    "refus_priorite_intersection",
]


def build_scenario(name: str, variation: int = 0) -> Scenario:
    """Construit un scénario nommé, avec une variation déterministe.

    Deux appels avec le même ``(name, variation)`` donnent exactement la même
    scène : le dataset est reproductible.
    """
    builder = SCENARIO_BUILDERS.get(name)
    if builder is None:
        raise KeyError(
            f"Scénario inconnu : {name!r}. Disponibles : {sorted(SCENARIO_BUILDERS)}"
        )
    return builder(random.Random(f"{name}:{variation}"))


def footprints_overlap(
    ax: float, ay: float, a_w: float, a_l: float,
    bx: float, by: float, b_w: float, b_l: float,
) -> bool:
    """Recouvrement des emprises au sol (rectangles alignés)."""
    return abs(ax - bx) < (a_w + b_w) * 0.5 and abs(ay - by) < (a_l + b_l) * 0.5
