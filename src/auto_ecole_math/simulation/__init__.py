"""Exports du sous-module simulation (génération de séquences annotées)."""

from .actors import ACTOR_CLASSES, Actor, ActorClass, Ego, LaneChange, Phase
from .generator import DatasetGenerator, GeneratorConfig, GeneratedSequence
from .renderer import DashcamRenderer, RenderConfig
from .scenarios import ERROR_LABELS, SCENARIO_BUILDERS, Scenario, build_scenario

__all__ = [
    "ACTOR_CLASSES",
    "ERROR_LABELS",
    "SCENARIO_BUILDERS",
    "Actor",
    "ActorClass",
    "DashcamRenderer",
    "DatasetGenerator",
    "Ego",
    "GeneratedSequence",
    "GeneratorConfig",
    "LaneChange",
    "Phase",
    "RenderConfig",
    "Scenario",
    "build_scenario",
]
