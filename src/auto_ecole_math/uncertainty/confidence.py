"""Lissage temporel et analyse bayésienne des scores de confiance YOLO.

1. **Lissage exponentiel (EWMA)**

.. math::

    s_t = \\alpha\\, c_t + (1 - \\alpha)\\, s_{t-1}

2. **Modèle Beta–Bernoulli** : en traitant chaque détection au-dessus
d'un seuil comme un succès bernoullien, la postérieure

.. math::

    p(\\theta \\mid D) = \\mathrm{Beta}(\\alpha_0 + s,\\, \\beta_0 + f)

fournit moyenne et variance sur la « vraie » présence de l'objet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExponentialSmoother:
    """Lisseur exponentiel sur un score scalaire :math:`c_t \\in [0, 1]`.

    Parameters
    ----------
    alpha :
        Facteur d'oubli :math:`\\alpha \\in (0, 1]` (plus grand = plus réactif).
    """

    alpha: float = 0.3
    value: float | None = None

    def update(self, confidence: float) -> float:
        """Met à jour :math:`s_t` et retourne la valeur lissée."""
        c = float(confidence)
        if self.value is None:
            self.value = c
        else:
            a = float(self.alpha)
            self.value = a * c + (1.0 - a) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class BetaBernoulliConfidence:
    """Inférence bayésienne Beta–Bernoulli sur la présence d'un objet.

    Prior :math:`\\mathrm{Beta}(\\alpha_0, \\beta_0)`. À chaque frame :

    - succès si :math:`c_t \\ge \\tau` → :math:`\\alpha \\leftarrow \\alpha + 1`
    - échec sinon → :math:`\\beta \\leftarrow \\beta + 1`

    La moyenne postérieure est :math:`\\mathbb{E}[\\theta] = \\frac{\\alpha}{\\alpha+\\beta}`
    et la variance
    :math:`\\mathrm{Var}(\\theta) = \\frac{\\alpha\\beta}{(\\alpha+\\beta)^2(\\alpha+\\beta+1)}`.
    """

    alpha0: float = 1.0
    beta0: float = 1.0
    threshold: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha0)
        self.beta = float(self.beta0)

    def update(self, confidence: float) -> dict[str, float]:
        """Met à jour la postérieure et retourne moyenne / variance / IC approximatif."""
        if float(confidence) >= self.threshold:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        return self.summary()

    def summary(self) -> dict[str, float]:
        """Statistiques de la loi Beta courante."""
        a, b = self.alpha, self.beta
        total = a + b
        mean = a / total
        var = (a * b) / (total * total * (total + 1.0))
        # Approximate 95% interval via normal (adequate for large counts)
        std = var**0.5
        return {
            "posterior_mean": mean,
            "posterior_var": var,
            "posterior_std": std,
            "ci95_low": max(0.0, mean - 1.96 * std),
            "ci95_high": min(1.0, mean + 1.96 * std),
            "alpha": a,
            "beta": b,
        }

    def reset(self) -> None:
        self.alpha = float(self.alpha0)
        self.beta = float(self.beta0)


class TrackConfidenceFilter:
    """Combine EWMA + Beta–Bernoulli par ``track_id``."""

    def __init__(self, alpha: float = 0.3, beta_threshold: float = 0.5) -> None:
        self.alpha = alpha
        self.beta_threshold = beta_threshold
        self._ewma: dict[int, ExponentialSmoother] = {}
        self._beta: dict[int, BetaBernoulliConfidence] = {}

    def update(self, track_id: int, confidence: float) -> dict[str, float]:
        if track_id not in self._ewma:
            self._ewma[track_id] = ExponentialSmoother(alpha=self.alpha)
            self._beta[track_id] = BetaBernoulliConfidence(threshold=self.beta_threshold)
        smoothed = self._ewma[track_id].update(confidence)
        bayes = self._beta[track_id].update(confidence)
        return {"smoothed_conf": smoothed, **bayes}
