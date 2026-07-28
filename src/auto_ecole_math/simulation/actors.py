"""Modèle cinématique des acteurs de la scène simulée.

Tout est exprimé dans un repère **monde** plan (mètres) :

- ``X`` : axe latéral, positif vers la droite de la route
- ``Y`` : axe longitudinal, positif dans le sens de circulation

Le repère **sol ego-relatif** utilisé par le pipeline s'en déduit par simple
translation : :math:`X_{rel} = X - X_{ego}`, :math:`Y_{rel} = Y - Y_{ego}`.

Chaque acteur suit un mouvement à accélération constante par morceaux
(*phases*), auquel peut s'ajouter un changement de voie interpolé en
*smoothstep*. C'est suffisant pour décrire freinage tardif, déboîtement,
traversée de piéton — et cela reste analytiquement intégrable, donc la vérité
terrain est exacte (pas d'erreur d'intégration numérique accumulée).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActorClass:
    """Gabarit physique et rendu d'une catégorie d'objet.

    ``label`` reprend le vocabulaire COCO utilisé par YOLO afin que les
    annotations générées soient directement comparables aux détections.
    """

    label: str
    width_m: float
    length_m: float
    height_m: float
    color: tuple[int, int, int]  # BGR (OpenCV)


#: Gabarits standards (dimensions moyennes réelles)
ACTOR_CLASSES: dict[str, ActorClass] = {
    "car": ActorClass("car", 1.8, 4.5, 1.5, (150, 120, 90)),
    "truck": ActorClass("truck", 2.5, 8.0, 3.2, (120, 120, 150)),
    "bus": ActorClass("bus", 2.5, 11.0, 3.2, (90, 140, 170)),
    "motorcycle": ActorClass("motorcycle", 0.8, 2.1, 1.5, (110, 90, 160)),
    "bicycle": ActorClass("bicycle", 0.6, 1.7, 1.8, (80, 160, 120)),
    "person": ActorClass("person", 0.6, 0.6, 1.7, (90, 90, 190)),
}


@dataclass(frozen=True)
class Phase:
    """Segment de mouvement à accélération constante."""

    duration_s: float
    ax_mps2: float = 0.0
    ay_mps2: float = 0.0


@dataclass(frozen=True)
class LaneChange:
    """Déport latéral progressif (interpolation *smoothstep*).

    Modélise un déboîtement : le véhicule glisse de ``dx_m`` mètres entre
    ``t_start_s`` et ``t_end_s`` sans à-coup de vitesse.
    """

    t_start_s: float
    t_end_s: float
    dx_m: float


def _smoothstep(t: float) -> float:
    """Interpolation douce :math:`3t^2 - 2t^3` sur :math:`[0, 1]`."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_derivative(t: float) -> float:
    """Dérivée de la *smoothstep* : :math:`6t(1-t)`."""
    t = min(1.0, max(0.0, t))
    return 6.0 * t * (1.0 - t)


@dataclass
class Actor:
    """Un objet mobile de la scène (véhicule, piéton, cycliste).

    Parameters
    ----------
    actor_id :
        Identifiant stable, sert de vérité terrain pour le suivi.
    cls :
        Gabarit physique.
    x0_m, y0_m :
        Position monde à :math:`t = 0`.
    vx0_mps, vy0_mps :
        Vitesse initiale.
    phases :
        Suite de segments à accélération constante. Après la dernière phase le
        mouvement se poursuit à vitesse constante.
    lane_change :
        Déport latéral optionnel superposé au mouvement.
    t_appear_s, t_vanish_s :
        Fenêtre de présence dans la scène (un piéton peut surgir en cours de
        séquence).
    """

    actor_id: int
    cls: ActorClass
    x0_m: float
    y0_m: float
    vx0_mps: float = 0.0
    vy0_mps: float = 0.0
    phases: list[Phase] = field(default_factory=list)
    lane_change: LaneChange | None = None
    t_appear_s: float = 0.0
    t_vanish_s: float | None = None

    def is_present(self, t: float) -> bool:
        """L'acteur est-il visible dans la scène à l'instant ``t`` ?"""
        if t < self.t_appear_s:
            return False
        return self.t_vanish_s is None or t <= self.t_vanish_s

    def state(self, t: float) -> tuple[float, float, float, float]:
        """État exact ``(x, y, vx, vy)`` à l'instant ``t``.

        Intégration analytique des phases : aucune dérive numérique, la valeur
        retournée *est* la vérité terrain.
        """
        x, y = self.x0_m, self.y0_m
        vx, vy = self.vx0_mps, self.vy0_mps
        remaining = t

        for phase in self.phases:
            dt = min(remaining, phase.duration_s)
            if dt <= 0.0:
                break
            x += vx * dt + 0.5 * phase.ax_mps2 * dt * dt
            y += vy * dt + 0.5 * phase.ay_mps2 * dt * dt
            vx += phase.ax_mps2 * dt
            vy += phase.ay_mps2 * dt
            remaining -= dt
            if remaining <= 0.0:
                break

        if remaining > 0.0:  # au-delà des phases : vitesse constante
            x += vx * remaining
            y += vy * remaining

        if self.lane_change is not None:
            lc = self.lane_change
            span = max(lc.t_end_s - lc.t_start_s, 1e-9)
            u = (t - lc.t_start_s) / span
            x += lc.dx_m * _smoothstep(u)
            if 0.0 <= u <= 1.0:
                vx += lc.dx_m * _smoothstep_derivative(u) / span

        return x, y, vx, vy


@dataclass
class Ego:
    """Le véhicule de l'élève : porte la caméra, origine du repère sol."""

    x0_m: float = 0.0
    y0_m: float = 0.0
    speed_mps: float = 13.9  # ~50 km/h
    phases: list[Phase] = field(default_factory=list)
    lane_change: LaneChange | None = None

    def as_actor(self) -> Actor:
        """Vue ``Actor`` de l'ego, pour réutiliser la même intégration."""
        return Actor(
            actor_id=0,
            cls=ACTOR_CLASSES["car"],
            x0_m=self.x0_m,
            y0_m=self.y0_m,
            vx0_mps=0.0,
            vy0_mps=self.speed_mps,
            phases=self.phases,
            lane_change=self.lane_change,
        )

    def state(self, t: float) -> tuple[float, float, float, float]:
        """État exact ``(x, y, vx, vy)`` de l'ego."""
        return self.as_actor().state(t)
