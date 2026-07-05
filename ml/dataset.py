"""Build labeled feature datasets from synthetic sessions via the REAL pipeline.

This deliberately runs synthetic CSI through the exact same code that will
process real hardware captures: parse -> amplitude matrix -> preprocess ->
sliding-window feature extraction. That way the training data and the future
live data share one code path, and training-time bugs surface immediately.

For people counting we also build a two-node dataset: each window is
represented by features from node1 concatenated with features from node2,
matching the planned 2x RX-node deployment for spatial diversity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.synthetic import SCENARIOS, generate_session
from pipeline.config import PipelineConfig
from pipeline.features import FEATURE_COLUMNS, extract_features
from pipeline.parser import parse_csi_line
from pipeline.preprocess import preprocess


def _session_amplitude_via_parser(lines: list[str]) -> np.ndarray:
    """Parse raw synthetic lines and stack their amplitude vectors.

    Exercises pipeline.parser exactly as live serial data will.
    """
    amps = [parse_csi_line(ln).amplitude for ln in lines]
    return np.vstack(amps).astype(np.float32)


def build_single_node_dataset(
    config: PipelineConfig,
    sessions_per_scenario: int = 6,
    duration_s: float = 20.0,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Feature DataFrame with a `label` column, one row per window.

    Each synthetic session is parsed, preprocessed, and windowed. Scenarios
    map to labels empty/walking/fall/2people.
    """
    frames: list[pd.DataFrame] = []
    seed = base_seed
    for scenario in SCENARIOS:
        for _ in range(sessions_per_scenario):
            sess = generate_session(
                scenario, duration_s=duration_s,
                sample_rate_hz=config.window.sample_rate_hz, seed=seed,
            )
            seed += 1
            amp = _session_amplitude_via_parser(sess.lines)
            amp = preprocess(amp, config)
            feats = extract_features(amp, config, label=sess.label)
            frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def build_two_node_counting_dataset(
    config: PipelineConfig,
    sessions_per_count: int = 8,
    duration_s: float = 20.0,
    base_seed: int = 1000,
) -> pd.DataFrame:
    """Two-node dataset for 0/1/2 people counting.

    count 0 <- empty, count 1 <- walking, count 2 <- 2people. Two independent
    synthetic sessions (different seeds) stand in for the two RX nodes viewing
    the same scene from different positions. Features are concatenated with
    _n1 / _n2 suffixes.
    """
    count_to_scenario = {0: "empty", 1: "walking", 2: "2people"}
    rows: list[pd.DataFrame] = []
    seed = base_seed
    for count, scenario in count_to_scenario.items():
        for _ in range(sessions_per_count):
            n1 = generate_session(scenario, duration_s=duration_s,
                                  sample_rate_hz=config.window.sample_rate_hz, seed=seed)
            n2 = generate_session(scenario, duration_s=duration_s,
                                  sample_rate_hz=config.window.sample_rate_hz, seed=seed + 5000)
            seed += 1
            f1 = _windows_features(n1.lines, config).add_suffix("_n1")
            f2 = _windows_features(n2.lines, config).add_suffix("_n2")
            m = min(len(f1), len(f2))
            merged = pd.concat(
                [f1.iloc[:m].reset_index(drop=True), f2.iloc[:m].reset_index(drop=True)],
                axis=1,
            )
            merged["count"] = count
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _windows_features(lines: list[str], config: PipelineConfig) -> pd.DataFrame:
    amp = _session_amplitude_via_parser(lines)
    amp = preprocess(amp, config)
    feats = extract_features(amp, config)
    return feats[list(FEATURE_COLUMNS)]
