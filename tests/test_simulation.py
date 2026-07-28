"""Tests — génération de séquences annotées et cohérence de la caméra."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_ecole_math.geometry.homography import HomographyEstimator  # noqa: E402
from auto_ecole_math.simulation.actors import (  # noqa: E402
    ACTOR_CLASSES,
    Actor,
    Ego,
    LaneChange,
    Phase,
)
from auto_ecole_math.simulation.camera import (  # noqa: E402
    PinholeCamera,
    check_pinhole_consistency,
    lateral_scale_px_per_m,
)
from auto_ecole_math.simulation.generator import (  # noqa: E402
    DatasetGenerator,
    GeneratorConfig,
)
from auto_ecole_math.simulation.renderer import DashcamRenderer  # noqa: E402
from auto_ecole_math.simulation.scenarios import (  # noqa: E402
    SCENARIO_BUILDERS,
    Scenario,
    build_scenario,
    footprints_overlap,
)


# --- Cinématique -----------------------------------------------------------


def test_mouvement_uniforme():
    actor = Actor(1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=10.0, vy0_mps=20.0)
    x, y, vx, vy = actor.state(2.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(50.0)
    assert vy == pytest.approx(20.0)


def test_phase_de_freinage_exacte():
    """Intégration analytique : y = y0 + v t + a t²/2, v = v0 + a t."""
    actor = Actor(
        1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=0.0, vy0_mps=20.0,
        phases=[Phase(3.0, ay_mps2=-5.0)],
    )
    _, y, _, vy = actor.state(2.0)
    assert y == pytest.approx(20 * 2 - 0.5 * 5 * 4)  # 30 m
    assert vy == pytest.approx(10.0)


def test_mouvement_constant_apres_les_phases():
    actor = Actor(
        1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=0.0, vy0_mps=10.0,
        phases=[Phase(1.0, ay_mps2=10.0)],
    )
    # Après 1 s : y=15, v=20 ; puis vitesse constante
    _, y3, _, vy3 = actor.state(3.0)
    assert vy3 == pytest.approx(20.0)
    assert y3 == pytest.approx(15.0 + 20.0 * 2.0)


def test_changement_de_voie_complet_et_continu():
    actor = Actor(
        1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=0.0, vy0_mps=10.0,
        lane_change=LaneChange(1.0, 3.0, 3.5),
    )
    assert actor.state(0.5)[0] == pytest.approx(0.0)  # avant
    assert actor.state(2.0)[0] == pytest.approx(1.75)  # mi-parcours
    assert actor.state(4.0)[0] == pytest.approx(3.5)  # après
    # Vitesse latérale nulle aux extrémités : pas d'à-coup
    assert actor.state(1.0)[2] == pytest.approx(0.0, abs=1e-9)
    assert actor.state(3.0)[2] == pytest.approx(0.0, abs=1e-9)


def test_fenetre_de_presence():
    actor = Actor(1, ACTOR_CLASSES["person"], 0.0, 20.0, t_appear_s=2.0, t_vanish_s=5.0)
    assert not actor.is_present(1.0)
    assert actor.is_present(3.0)
    assert not actor.is_present(6.0)


# --- Caméra ----------------------------------------------------------------


def test_horizon_conforme_a_la_formule_stenope():
    cam = PinholeCamera(pitch_deg=4.0)
    _, cy = cam.principal_point
    attendu = cy - cam.focal_px * math.tan(math.radians(4.0))
    assert cam.horizon_v == pytest.approx(attendu)


def test_projection_converge_vers_l_horizon():
    cam = PinholeCamera()
    v_loin = cam.ground_to_pixel(0.0, 1e6)[1]
    assert v_loin == pytest.approx(cam.horizon_v, abs=1e-2)


def test_echelle_laterale_en_inverse_de_la_distance():
    """Un sténopé impose s(Y) ∝ 1/Y : c'est le test que rate la calibration livrée."""
    cam = PinholeCamera()
    calib = cam.calibration_dict()
    homography = _homography_from_dict(calib)
    s10 = lateral_scale_px_per_m(homography, 10.0)
    s30 = lateral_scale_px_per_m(homography, 30.0)
    assert s10 / s30 == pytest.approx(3.0, rel=0.05)


def test_calibration_synthetique_est_coherente():
    cam = PinholeCamera()
    homography = _homography_from_dict(cam.calibration_dict())
    report = check_pinhole_consistency(homography)
    assert report["consistent"]
    assert report["relative_error"] < 0.05


def test_calibration_livree_est_detectee_incoherente():
    """Garde-fou : ``homography_default.json`` n'est pas physiquement réalisable."""
    default_path = ROOT / "config" / "homography_default.json"
    if not default_path.exists():
        pytest.skip("calibration par défaut absente")
    homography = HomographyEstimator.from_json(default_path)
    report = check_pinhole_consistency(homography, y_near_m=4.0, y_far_m=20.0)
    assert not report["consistent"]
    # L'échelle indique un point lointain bien plus proche que les 20 m annoncés
    assert report["implied_far_y_m"] < 10.0


def test_renderer_et_camera_saccordent_sur_l_horizon():
    """Deux calculs indépendants de l'horizon doivent coïncider."""
    cam = PinholeCamera()
    homography = _homography_from_dict(cam.calibration_dict())
    renderer = DashcamRenderer(homography)
    assert renderer.horizon_v(cam.image_size[0] / 2) == pytest.approx(
        cam.horizon_v, abs=0.5
    )


# --- Scénarios -------------------------------------------------------------


def test_scenarios_reproductibles():
    a = build_scenario("pieton_non_anticipe", 3)
    b = build_scenario("pieton_non_anticipe", 3)
    assert a.ego.speed_mps == b.ego.speed_mps
    assert a.actors[0].y0_m == b.actors[0].y0_m


def test_variations_differentes():
    a = build_scenario("pieton_non_anticipe", 0)
    b = build_scenario("pieton_non_anticipe", 1)
    assert a.ego.speed_mps != b.ego.speed_mps


def test_scenario_inconnu_rejete():
    with pytest.raises(KeyError):
        build_scenario("scenario_qui_nexiste_pas")


def test_conduite_saine_sans_erreur():
    scenario = build_scenario("conduite_saine", 0)
    assert scenario.error_type is None
    assert not scenario.expect_collision


def test_tous_les_scenarios_ont_une_etiquette_coherente():
    for name in SCENARIO_BUILDERS:
        scenario = build_scenario(name, 0)
        if scenario.error_type is None:
            assert not scenario.expect_collision
        else:
            assert scenario.error_end_s >= scenario.error_start_s


def test_recouvrement_des_emprises():
    assert footprints_overlap(0, 0, 1.8, 4.5, 0, 3.0, 1.8, 4.5)
    assert not footprints_overlap(0, 0, 1.8, 4.5, 0, 6.0, 1.8, 4.5)
    assert not footprints_overlap(0, 0, 1.8, 4.5, 3.5, 0, 1.8, 4.5)


# --- Génération ------------------------------------------------------------


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Génère une courte séquence de collision, réutilisée par plusieurs tests."""
    out = tmp_path_factory.mktemp("dataset")
    generator = DatasetGenerator(GeneratorConfig(output_dir=out, fps=30.0))
    scenario = Scenario(
        name="test_court",
        description="Approche d'un véhicule à l'arrêt",
        duration_s=2.0,
        ego=Ego(speed_mps=12.0),
        actors=[Actor(1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=30.0, vy0_mps=0.0)],
        error_type="distance_securite_insuffisante",
        error_start_s=0.0,
        error_end_s=2.0,
        expect_collision=False,
    )
    return generator, generator.generate(scenario)


def test_video_lisible_et_de_bonne_taille(generated):
    _, seq = generated
    assert seq.video_path.exists() and seq.video_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(seq.video_path))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame.shape == (832, 1280, 3)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == seq.n_frames
    finally:
        cap.release()


def test_frames_non_uniformes(generated):
    """Une image constante signalerait un rendu muet."""
    _, seq = generated
    cap = cv2.VideoCapture(str(seq.video_path))
    try:
        _, frame = cap.read()
    finally:
        cap.release()
    assert frame.std() > 10.0


def test_metadonnees_completes(generated):
    _, seq = generated
    meta = json.loads(seq.meta_path.read_text(encoding="utf-8"))
    for key in ("scenario", "error_type", "fps", "n_frames", "calibration_path"):
        assert key in meta
    assert meta["n_frames"] == seq.n_frames


def test_verite_terrain_reprojetable(generated):
    """Le cœur du dispositif : bbox → homographie ⇒ position vraie.

    Le décalage attendu est la demi-longueur du véhicule, car la boîte touche
    le sol à son bord avant et non en son centre.
    """
    generator, seq = generated
    rows = list(csv.DictReader(seq.truth_path.open(encoding="utf-8")))
    assert rows

    half_len = ACTOR_CLASSES["car"].length_m * 0.5
    ecarts_y, ecarts_x = [], []
    for row in rows:
        bbox = [float(row[k]) for k in ("x1_px", "y1_px", "x2_px", "y2_px")]
        est = generator.homography.bbox_to_ground(bbox)
        ecarts_x.append(est[0] - float(row["X_m"]))
        ecarts_y.append(est[1] + half_len - float(row["Y_m"]))

    assert np.max(np.abs(ecarts_x)) < 0.05
    assert np.max(np.abs(ecarts_y)) < 0.30


def test_boites_dans_l_image(generated):
    generator, seq = generated
    w, h = generator.renderer.cfg.image_size
    rows = list(csv.DictReader(seq.truth_path.open(encoding="utf-8")))
    for row in rows:
        assert float(row["x2_px"]) > float(row["x1_px"])
        assert float(row["y2_px"]) > float(row["y1_px"])
        # Un objet devant l'ego reste au moins partiellement visible
        assert float(row["x2_px"]) > 0 and float(row["x1_px"]) < w
        assert float(row["y2_px"]) > 0 and float(row["y1_px"]) < h


def test_objet_grossit_en_se_rapprochant(generated):
    """Contrôle de perspective : l'aire de la boîte croît quand la distance chute."""
    _, seq = generated
    rows = list(csv.DictReader(seq.truth_path.open(encoding="utf-8")))
    premier, dernier = rows[0], rows[-1]

    def aire(row):
        return (float(row["x2_px"]) - float(row["x1_px"])) * (
            float(row["y2_px"]) - float(row["y1_px"])
        )

    assert float(dernier["d_m"]) < float(premier["d_m"])
    assert aire(dernier) > aire(premier)


def test_collision_detectee_et_sequence_tronquee(tmp_path):
    generator = DatasetGenerator(GeneratorConfig(output_dir=tmp_path, fps=30.0))
    scenario = Scenario(
        name="impact",
        description="Véhicule à l'arrêt percuté sans freinage",
        duration_s=10.0,
        ego=Ego(speed_mps=15.0),
        actors=[Actor(1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=20.0, vy0_mps=0.0)],
        expect_collision=True,
    )
    seq = generator.generate(scenario)
    assert seq.collision_frame is not None
    # La séquence s'arrête au contact, bien avant les 10 s demandées
    assert seq.n_frames == seq.collision_frame
    assert seq.n_frames < 10.0 * 30.0


def test_export_yolo(tmp_path):
    generator = DatasetGenerator(
        GeneratorConfig(output_dir=tmp_path, fps=30.0, export_yolo=True)
    )
    scenario = Scenario(
        name="yolo",
        description="Suivi simple",
        duration_s=1.0,
        ego=Ego(speed_mps=10.0),
        actors=[Actor(1, ACTOR_CLASSES["car"], x0_m=0.0, y0_m=25.0, vy0_mps=10.0)],
    )
    seq = generator.generate(scenario)
    labels_dir = tmp_path / "labels" / seq.name
    fichiers = sorted(labels_dir.glob("frame_*.txt"))
    assert fichiers
    assert (labels_dir / "classes.txt").exists()

    champs = fichiers[0].read_text(encoding="utf-8").split()
    assert len(champs) == 5  # classe cx cy w h
    assert all(0.0 <= float(v) <= 1.0 for v in champs[1:])


def _homography_from_dict(calib: dict) -> HomographyEstimator:
    """Construit un estimateur depuis un dictionnaire de calibration en mémoire."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(calib, f)
        path = f.name
    return HomographyEstimator.from_json(path)
