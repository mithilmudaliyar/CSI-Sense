"""Sliding-window feature extraction for CSI human-sensing.

Input is a denoised amplitude matrix of shape (n_samples, n_subcarriers)
plus a nominal sample rate. Output is a tidy pandas DataFrame with one
row per window, ready for scikit-learn.

The features are deliberately interpretable so they can be reasoned about
for presence, fall, and people-counting tasks:

- amp_var / amp_std      : overall motion energy (presence signal)
- amp_mean_abs_diff      : average rate-of-change between consecutive samples
- amp_max_abs_diff       : the single sharpest jump (fall spike signal)
- amp_entropy            : spectral/temporal disorder (walking vs still)
- n_disturbance_peaks    : count of distinct motion bursts in the window
- post_event_stillness_s : seconds of near-stillness AFTER the biggest spike
                           (a fall = big spike THEN sudden stillness)
- amp_energy_ratio       : fraction of window energy in its most active half
- corr_across_subcarriers: mean pairwise correlation across subcarriers
                           (bodies move many subcarriers together)

Everything is computed from a collapsed per-sample signal (mean across
subcarriers) plus a few cross-subcarrier statistics, so the feature count
is independent of how many subcarriers survive null-removal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig

FEATURE_COLUMNS = (
    "amp_var",
    "amp_std",
    "amp_mean_abs_diff",
    "amp_max_abs_diff",
    "amp_entropy",
    "n_disturbance_peaks",
    "post_event_stillness_s",
    "amp_energy_ratio",
    "corr_across_subcarriers",
    "subcarrier_var_mean",
)

# Fraction of a window's peak-to-peak range above which we count a
# "disturbance peak". Named constant rather than a magic number.
PEAK_PROMINENCE_FRAC = 0.3
# Rate-of-change below this fraction of the window's std counts as "still".
STILLNESS_FRAC = 0.15


def _signal_entropy(sig: np.ndarray, n_bins: int = 16) -> float:
    """Shannon entropy of the amplitude histogram (normalised 0..1)."""
    if sig.size < 2:
        return 0.0
    lo, hi = float(sig.min()), float(sig.max())
    if hi - lo < 1e-9:
        return 0.0
    hist, _ = np.histogram(sig, bins=n_bins, range=(lo, hi))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    ent = -np.sum(p * np.log2(p))
    return float(ent / np.log2(n_bins))  # normalise by max possible entropy


def _count_disturbance_peaks(diff: np.ndarray) -> int:
    """Count distinct bursts where |rate-of-change| crosses a prominence gate."""
    if diff.size == 0:
        return 0
    mag = np.abs(diff)
    span = float(mag.max())
    if span < 1e-9:
        return 0
    gate = PEAK_PROMINENCE_FRAC * span
    above = mag > gate
    # Count rising edges (transitions from below-gate to above-gate).
    rising = np.logical_and(above[1:], ~above[:-1])
    return int(rising.sum()) + int(above[0])


def _post_event_stillness_seconds(sig: np.ndarray, dt: float) -> float:
    """Seconds of near-stillness immediately after the window's sharpest change.

    Fall signature: a large amplitude spike followed by the person lying
    motionless. We locate the biggest sample-to-sample jump, then measure
    how long the rate-of-change stays below a stillness gate afterwards.
    """
    if sig.size < 3:
        return 0.0
    diff = np.abs(np.diff(sig))
    if diff.size == 0:
        return 0.0
    spike_idx = int(np.argmax(diff))
    tail = diff[spike_idx + 1 :]
    if tail.size == 0:
        return 0.0
    gate = STILLNESS_FRAC * float(sig.std() + 1e-9)
    still = 0
    for v in tail:
        if v <= gate:
            still += 1
        else:
            break
    return float(still * dt)


def _energy_ratio(sig: np.ndarray) -> float:
    """Fraction of the window's variance concentrated in its more active half."""
    if sig.size < 2:
        return 0.0
    mid = sig.size // 2
    first = sig[:mid]
    second = sig[mid:]
    e1 = float(first.var()) if first.size else 0.0
    e2 = float(second.var()) if second.size else 0.0
    total = e1 + e2
    if total < 1e-12:
        return 0.5
    return max(e1, e2) / total


def _mean_cross_correlation(window: np.ndarray, max_sub: int = 12) -> float:
    """Mean absolute pairwise correlation across a subset of subcarriers.

    Coordinated body motion moves many subcarriers together, raising this;
    independent electronic noise keeps it near zero. Subsampled to keep it
    cheap and stable when many subcarriers survive.
    """
    if window.ndim != 2 or window.shape[1] < 2:
        return 0.0
    n_sub = window.shape[1]
    idx = np.linspace(0, n_sub - 1, min(max_sub, n_sub)).astype(int)
    sub = window[:, np.unique(idx)]
    if np.allclose(sub.std(axis=0), 0):
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(sub, rowvar=False)
    if c.ndim != 2:
        return 0.0
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals)))


def window_features(window: np.ndarray, sample_rate_hz: float) -> dict:
    """Compute the feature dict for a single (n_samples, n_subcarriers) window."""
    if window.ndim == 1:
        window = window[:, None]
    dt = 1.0 / sample_rate_hz if sample_rate_hz > 0 else 1.0
    # Collapse subcarriers to a per-sample motion signal.
    sig = window.mean(axis=1).astype(np.float64)
    diff = np.diff(sig)

    return {
        "amp_var": float(sig.var()),
        "amp_std": float(sig.std()),
        "amp_mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "amp_max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "amp_entropy": _signal_entropy(sig),
        "n_disturbance_peaks": _count_disturbance_peaks(diff),
        "post_event_stillness_s": _post_event_stillness_seconds(sig, dt),
        "amp_energy_ratio": _energy_ratio(sig),
        "corr_across_subcarriers": _mean_cross_correlation(window),
        "subcarrier_var_mean": float(np.mean(window.var(axis=0))),
    }


def extract_features(
    amplitude: np.ndarray,
    config: PipelineConfig,
    label: str | None = None,
    node_id: str = "node1",
) -> pd.DataFrame:
    """Slide a window over an amplitude matrix and produce a feature DataFrame.

    One row per window. If ``label`` is given it is attached as a column so
    the frame can be concatenated straight into a training set.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim == 1:
        amplitude = amplitude[:, None]
    n_samples = amplitude.shape[0]
    win = config.window.window_samples
    hop = config.window.hop_samples
    sr = config.window.sample_rate_hz

    rows: list[dict] = []
    if n_samples < win:
        # Too short for a full window: emit a single padded-by-availability window.
        starts = [0]
        win_eff = n_samples
    else:
        starts = list(range(0, n_samples - win + 1, hop))
        win_eff = win

    for start in starts:
        window = amplitude[start : start + win_eff]
        feats = window_features(window, sr)
        feats["window_start_sample"] = start
        feats["window_start_s"] = start / sr if sr > 0 else float(start)
        feats["node_id"] = node_id
        if label is not None:
            feats["label"] = label
        rows.append(feats)

    return pd.DataFrame(rows)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the numeric feature matrix (in canonical column order)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing}")
    return df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
