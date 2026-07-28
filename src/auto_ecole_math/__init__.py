"""Auto-École Math — perception mathématique de la conduite.

Modules
-------
- ``geometry`` : homographie / IPM pixels → mètres
- ``tracking`` : filtre de Kalman CV + multi-object tracking
- ``kinematics`` : TTC et distance de freinage
- ``uncertainty`` : lissage de confiance et stats de séance
- ``pipeline`` : orchestration YOLO → rapport quantitatif
"""

from auto_ecole_math.geometry import HomographyEstimator, load_calibration
from auto_ecole_math.kinematics import braking_distance, time_to_collision
from auto_ecole_math.tracking import KalmanCV2D, MultiObjectTracker
from auto_ecole_math.uncertainty import SessionStatsCollector, TrackConfidenceFilter

__all__ = [
    "HomographyEstimator",
    "load_calibration",
    "KalmanCV2D",
    "MultiObjectTracker",
    "time_to_collision",
    "braking_distance",
    "TrackConfidenceFilter",
    "SessionStatsCollector",
]

__version__ = "0.1.0"
