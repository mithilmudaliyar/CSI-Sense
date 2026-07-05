"""Synthetic CSI generator for the four target scenarios.

Purpose: validate the ENTIRE pipeline (parse -> store -> preprocess ->
features -> classify) before any real hardware exists. This is NOT a
physically accurate WiFi channel model - it reproduces the *coarse
statistical signatures* each activity leaves in CSI amplitude so the
software can be exercised and unit-tested:

    empty    : low-variance noise around a static multipath baseline
    walking  : baseline + moderate quasi-periodic gait modulation
    fall     : near-still, then one sharp large spike, then sudden stillness
    2people  : superposition of two gait patterns -> higher variance,
               more disturbance peaks, stronger cross-subcarrier coupling

Each scenario can be emitted either as:
  - a raw list of ESP32-CSI-Tool serial lines (exercises the real parser), or
  - a (n_samples, n_subcarriers) amplitude matrix (convenient for ML).

The raw-line path quantizes to int8 [imag, real] pairs exactly like the
firmware buffer, so pipeline.parser round-trips it faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# LLTF-only capture: 64 complex subcarriers -> len=128 int8 values.
# (Matches ESP32-CSI-Tool with CONFIG_SHOULD_COLLECT_ONLY_LLTF.)
N_SUBCARRIERS = 64
BUF_LEN = N_SUBCARRIERS * 2  # int8 count reported in the `len` field
# LLTF null/guard subcarrier slots (DC + band edges) - kept ~zero so the
# preprocessing null-subcarrier removal has something real to drop.
NULL_SUBCARRIERS = tuple([0, 1, 2, 3, 4, 5, 31, 32, 59, 60, 61, 62, 63])

SCENARIOS = ("empty", "walking", "fall", "2people")
SCENARIO_TO_LABEL = {
    "empty": "empty",
    "walking": "walking",
    "fall": "fall",
    "2people": "2people",
}

DEFAULT_SAMPLE_RATE_HZ = 50.0
_INT8_CLIP = 127.0


@dataclass(frozen=True)
class SyntheticSession:
    scenario: str
    label: str
    amplitude: np.ndarray  # (n_samples, n_subcarriers), pre-quantization "truth"
    lines: list[str]       # raw ESP32-CSI-Tool serial lines
    sample_rate_hz: float


def _baseline_profile(rng: np.random.Generator) -> np.ndarray:
    """A static per-subcarrier multipath amplitude baseline (frequency-selective)."""
    sub = np.arange(N_SUBCARRIERS)
    # Smooth frequency-selective fading shape + mild ripple.
    base = 30.0 + 12.0 * np.sin(2 * np.pi * sub / N_SUBCARRIERS + rng.uniform(0, np.pi))
    base += 4.0 * np.sin(2 * np.pi * 3 * sub / N_SUBCARRIERS + rng.uniform(0, np.pi))
    base = np.clip(base, 6.0, 90.0)
    mask = np.ones(N_SUBCARRIERS)
    for k in NULL_SUBCARRIERS:
        mask[k] = 0.0
    return base * mask


def _empty(n: int, rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    noise = rng.normal(0.0, 0.6, size=(n, N_SUBCARRIERS))
    amp = base[None, :] + noise
    return amp


def _walking(
    n: int, rng: np.random.Generator, base: np.ndarray, sr: float,
    gait_hz: float | None = None, depth: float = 6.0, phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(n) / sr
    gait_hz = gait_hz if gait_hz is not None else rng.uniform(1.4, 2.2)
    # Subcarriers respond with different gains/phases to the moving body.
    sub_gain = rng.uniform(0.3, 1.0, size=N_SUBCARRIERS)
    sub_phase = rng.uniform(0, 2 * np.pi, size=N_SUBCARRIERS)
    modulation = depth * np.sin(2 * np.pi * gait_hz * t[:, None] + sub_phase[None, :] + phase)
    modulation *= sub_gain[None, :]
    # Slow drift (person moving through the room changes multipath).
    drift = 2.0 * np.sin(2 * np.pi * 0.15 * t)[:, None]
    noise = rng.normal(0.0, 0.8, size=(n, N_SUBCARRIERS))
    amp = base[None, :] + modulation + drift + noise
    # Zero-out null subcarriers again (modulation would otherwise fill them).
    for k in NULL_SUBCARRIERS:
        amp[:, k] = rng.normal(0.0, 0.4, size=n)
    return amp


def _fall(n: int, rng: np.random.Generator, base: np.ndarray, sr: float) -> np.ndarray:
    """Approach (mild motion) -> sharp spike -> sudden prolonged stillness."""
    amp = _empty(n, rng, base)
    # Mild motion in the first ~40% (person walking toward fall point).
    approach_end = int(0.4 * n)
    if approach_end > 5:
        amp[:approach_end] = _walking(
            approach_end, rng, base, sr, gait_hz=1.6, depth=4.0
        )
    # The fall: a sharp, large, short-lived disturbance.
    spike_idx = approach_end + max(1, int(0.02 * n))
    spike_len = max(2, int(0.05 * n))
    for j in range(spike_len):
        idx = spike_idx + j
        if idx >= n:
            break
        decay = 1.0 - j / spike_len
        amp[idx] += rng.normal(0.0, 3.0, size=N_SUBCARRIERS) + 45.0 * decay
    # After the spike: near-total stillness (person on the floor, motionless).
    still_start = spike_idx + spike_len
    if still_start < n:
        amp[still_start:] = base[None, :] + rng.normal(
            0.0, 0.3, size=(n - still_start, N_SUBCARRIERS)
        )
    for k in NULL_SUBCARRIERS:
        amp[:, k] = rng.normal(0.0, 0.4, size=n)
    return amp


def _two_people(n: int, rng: np.random.Generator, base: np.ndarray, sr: float) -> np.ndarray:
    w1 = _walking(n, rng, base, sr, gait_hz=rng.uniform(1.4, 1.8), depth=5.5, phase=0.0)
    w2 = _walking(n, rng, base, sr, gait_hz=rng.uniform(1.9, 2.4), depth=5.5,
                  phase=rng.uniform(0, np.pi))
    # Superposition around a single shared baseline (don't double-count base).
    amp = base[None, :] + (w1 - base[None, :]) + (w2 - base[None, :])
    amp += rng.normal(0.0, 0.9, size=(n, N_SUBCARRIERS))
    for k in NULL_SUBCARRIERS:
        amp[:, k] = rng.normal(0.0, 0.4, size=n)
    return amp


_GENERATORS = {
    "empty": lambda n, rng, base, sr: _empty(n, rng, base),
    "walking": lambda n, rng, base, sr: _walking(n, rng, base, sr),
    "fall": _fall,
    "2people": _two_people,
}


def _amplitude_to_lines(
    amp: np.ndarray, rng: np.random.Generator, role: str = "STA",
    mac: str = "AA:BB:CC:DD:EE:FF", channel: int = 6,
) -> list[str]:
    """Quantize an amplitude matrix into ESP32-CSI-Tool-format serial lines.

    We synthesize a plausible phase per subcarrier, convert to real/imag,
    clip to int8, and lay them out as [imag, real] pairs exactly like the
    firmware's raw buffer so pipeline.parser handles them unmodified.
    """
    n = amp.shape[0]
    lines: list[str] = []
    phase = rng.uniform(-np.pi, np.pi, size=amp.shape[1])
    local_ts = int(rng.integers(1_000_000, 5_000_000))
    for i in range(n):
        a = amp[i]
        real = np.clip(a * np.cos(phase), -_INT8_CLIP, _INT8_CLIP).astype(int)
        imag = np.clip(a * np.sin(phase), -_INT8_CLIP, _INT8_CLIP).astype(int)
        buf = np.empty(amp.shape[1] * 2, dtype=int)
        buf[0::2] = imag  # ESP-IDF layout: imaginary first
        buf[1::2] = real
        local_ts += int(20000 + rng.integers(-2000, 2000))  # ~50 Hz jittered
        real_ts = local_ts / 1_000_000.0
        buf_str = " ".join(str(int(v)) for v in buf) + " "
        line = (
            f"CSI_DATA,{role},{mac},-40,11,1,6,1,1,1,0,0,0,1,-92,0,"
            f"{channel},0,{local_ts},0,0,0,0,{real_ts:.6f},{len(buf)},[{buf_str}]"
        )
        lines.append(line)
    return lines


def generate_session(
    scenario: str,
    duration_s: float = 20.0,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    seed: int | None = None,
    node_id: str = "node1",
) -> SyntheticSession:
    """Generate one synthetic session for a scenario."""
    if scenario not in _GENERATORS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = max(4, int(round(duration_s * sample_rate_hz)))
    base = _baseline_profile(rng)
    amp = _GENERATORS[scenario](n, rng, base, sample_rate_hz)
    amp = np.clip(amp, 0.0, _INT8_CLIP)
    lines = _amplitude_to_lines(amp, rng)
    return SyntheticSession(
        scenario=scenario,
        label=SCENARIO_TO_LABEL[scenario],
        amplitude=amp.astype(np.float32),
        lines=lines,
        sample_rate_hz=sample_rate_hz,
    )
