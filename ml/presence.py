"""Presence / motion detector.

The easiest and most reliable task: is anyone moving in the space?
Motion inflates CSI amplitude variance, so a logistic regression on the
variance-family features is both effective and interpretable. A pure
threshold fallback is also provided for a zero-training deployment.

Labels are collapsed to binary: empty -> 0 (no presence), everything
else (walking/fall/2people) -> 1 (presence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.features import FEATURE_COLUMNS

PRESENCE_FEATURES = (
    "amp_var",
    "amp_std",
    "amp_mean_abs_diff",
    "amp_entropy",
    "corr_across_subcarriers",
    "subcarrier_var_mean",
)


def to_presence_labels(labels: pd.Series) -> np.ndarray:
    """Collapse activity labels to binary presence (empty=0, else=1)."""
    return (labels.astype(str) != "empty").astype(int).to_numpy()


def build_presence_model() -> Pipeline:
    """Standardized logistic regression on variance-family features."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


@dataclass
class ThresholdPresenceDetector:
    """Training-free fallback: presence if amplitude variance exceeds a gate.

    Useful on day one before any model is trained. The gate is calibrated
    from a short empty-room recording (mean + k*std of window variance).
    """

    variance_gate: float

    @classmethod
    def calibrate(cls, empty_amp_var: np.ndarray, k: float = 4.0) -> "ThresholdPresenceDetector":
        mu = float(np.mean(empty_amp_var))
        sd = float(np.std(empty_amp_var))
        return cls(variance_gate=mu + k * sd)

    def predict(self, amp_var: np.ndarray) -> np.ndarray:
        return (np.asarray(amp_var) > self.variance_gate).astype(int)


def presence_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in PRESENCE_FEATURES if c in df.columns]
    if not cols:
        raise ValueError(f"None of {PRESENCE_FEATURES} present in dataframe")
    return df[cols].to_numpy(dtype=np.float64)


__all__ = [
    "PRESENCE_FEATURES",
    "FEATURE_COLUMNS",
    "to_presence_labels",
    "build_presence_model",
    "ThresholdPresenceDetector",
    "presence_feature_matrix",
]
