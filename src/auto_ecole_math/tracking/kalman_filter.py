"""Filtre de Kalman linéaire à vitesse constante (Constant Velocity) en 2D.

État (plan sol, mètres) :

.. math::

    \\mathbf{z}_k = [X,\\, Y,\\, \\dot X,\\, \\dot Y]^\\top

Dynamique :

.. math::

    \\mathbf{z}_{k|k-1} = F(\\Delta t)\\, \\mathbf{z}_{k-1} + \\mathbf{w}_k,
    \\qquad
    F(\\Delta t) =
    \\begin{bmatrix}
    I_2 & \\Delta t\\, I_2 \\\\
    0   & I_2
    \\end{bmatrix}

Observation de position :

.. math::

    \\mathbf{y}_k = [X, Y]^\\top = H_{\\mathrm{obs}}\\, \\mathbf{z}_k + \\mathbf{v}_k,
    \\qquad
    H_{\\mathrm{obs}} = [I_2\\; 0]
"""

from __future__ import annotations

import numpy as np


class KalmanCV2D:
    """Filtre de Kalman Constant-Velocity 2D (état position + vitesse).

    Parameters
    ----------
    dt :
        Pas de temps initial :math:`\\Delta t` (s).
    process_var :
        Variance du bruit de process sur l'accélération
        (:math:`q`, m²/s⁴) — modèle de white-noise acceleration.
    meas_var :
        Variance du bruit de mesure sur :math:`(X, Y)` (m²).
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_var: float = 1.0,
        meas_var: float = 0.5,
    ) -> None:
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)

        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.R = np.eye(2, dtype=np.float64) * self.meas_var
        self._update_matrices(self.dt)
        self.initialized = False

    def _update_matrices(self, dt: float) -> None:
        """Met à jour :math:`F` et :math:`Q` pour un :math:`\\Delta t` donné."""
        dt = float(dt)
        self.dt = dt
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Discrete white-noise acceleration model (Bar-Shalom)
        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = 0.25 * dt4 * q
        q13 = 0.5 * dt3 * q
        q33 = dt2 * q
        self.Q = np.array(
            [
                [q11, 0.0, q13, 0.0],
                [0.0, q11, 0.0, q13],
                [q13, 0.0, q33, 0.0],
                [0.0, q13, 0.0, q33],
            ],
            dtype=np.float64,
        )

    def initiate(self, measurement: np.ndarray) -> None:
        """Initialise l'état avec une première mesure :math:`(X, Y)`."""
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        self.x[:] = 0.0
        self.x[0] = z[0]
        self.x[1] = z[1]
        self.P = np.diag([1.0, 1.0, 25.0, 25.0]).astype(np.float64)
        self.initialized = True

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Étape de prédiction :math:`\\mathbf{z}_{k|k-1} = F \\mathbf{z}_{k-1}`.

        Returns
        -------
        np.ndarray
            État prédit de dimension 4.
        """
        if dt is not None and abs(float(dt) - self.dt) > 1e-12:
            self._update_matrices(float(dt))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Étape de correction (mesure de position).

        .. math::

            K = P H^\\top (H P H^\\top + R)^{-1},\\;
            \\mathbf{z} \\leftarrow \\mathbf{z} + K(\\mathbf{y} - H\\mathbf{z})
        """
        if not self.initialized:
            self.initiate(measurement)
            return self.x.copy()

        y = np.asarray(measurement, dtype=np.float64).reshape(2)
        innov = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I = np.eye(4)
        # Joseph form for numerical stability
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()

    def mahalanobis(self, measurement: np.ndarray) -> float:
        """Distance de Mahalanobis entre prédiction et mesure.

        .. math::

            d_M = \\sqrt{ \\mathbf{e}^\\top S^{-1} \\mathbf{e} },
            \\quad \\mathbf{e} = \\mathbf{y} - H\\mathbf{z},\\;
            S = H P H^\\top + R
        """
        y = np.asarray(measurement, dtype=np.float64).reshape(2)
        innov = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(innov.T @ np.linalg.solve(S, innov)))

    @property
    def position(self) -> np.ndarray:
        """Position filtrée :math:`(X, Y)`."""
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Vitesse filtrée :math:`(\\dot X, \\dot Y)` en m/s."""
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        """Norme de la vitesse :math:`\\|\\mathbf{v}\\|_2` (m/s)."""
        return float(np.linalg.norm(self.velocity))
