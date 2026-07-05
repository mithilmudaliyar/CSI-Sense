"""Composable CSI denoising: Hampel (manual), PCA, wavelet.

All operate on an amplitude matrix of shape (n_samples, n_subcarriers)
and are toggled via PipelineConfig — nothing is hardcoded. Every
function returns a NEW array (no in-place mutation).
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def hampel_filter(x: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Manual Hampel outlier filter along axis 0 (time), per subcarrier.

    For each sample, compute the median and MAD over a window of
    ``window_size`` samples on each side; replace values more than
    ``n_sigmas`` scaled-MADs from the median with the window median.

    Implemented from scratch (no library) as a project requirement.
    """
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    n = x.shape[0]
    k = 1.4826  # MAD -> std scale factor for Gaussian data
    out = x.copy()

    # Build a (n, 2*window_size+1) sliding view per column via padding.
    pad = window_size
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    # windows[i] covers samples i-pad .. i+pad of the original signal
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * pad + 1, axis=0
    )  # shape (n, n_sub, 2*pad+1)
    med = np.median(windows, axis=-1)                    # (n, n_sub)
    mad = np.median(np.abs(windows - med[..., None]), axis=-1)
    threshold = n_sigmas * k * mad
    mask = np.abs(x - med) > threshold
    out[mask] = med[mask]

    n_replaced = int(mask.sum())
    if n_replaced:
        logger.debug("Hampel replaced %d/%d values", n_replaced, x.size)
    return out[:, 0] if squeeze else out


def pca_denoise(x: np.ndarray, n_components: int = 5) -> np.ndarray:
    """Denoise by projecting onto the top principal components and back.

    Keeps the dominant correlated motion structure across subcarriers and
    discards per-subcarrier independent noise.
    """
    if x.ndim != 2:
        raise ValueError("pca_denoise expects (n_samples, n_subcarriers)")
    n_components = min(n_components, min(x.shape))
    mean = x.mean(axis=0)
    centered = x - mean
    # SVD-based PCA (deterministic, no sklearn dependency here)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    s_trunc = np.zeros_like(s)
    s_trunc[:n_components] = s[:n_components]
    recon = (u * s_trunc) @ vt + mean
    return recon.astype(x.dtype)


def wavelet_denoise(
    x: np.ndarray, wavelet: str = "db4", level: int = 3, mode: str = "soft"
) -> np.ndarray:
    """Wavelet shrinkage denoising per subcarrier (uses PyWavelets)."""
    import pywt

    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    out = np.empty_like(x, dtype=np.float64)
    for col in range(x.shape[1]):
        sig = x[:, col].astype(np.float64)
        max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
        lvl = min(level, max_level) if max_level > 0 else 1
        coeffs = pywt.wavedec(sig, wavelet, level=lvl)
        # Universal threshold from the finest detail coefficients
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
        uthresh = sigma * np.sqrt(2 * np.log(max(len(sig), 2)))
        denoised = [coeffs[0]] + [
            pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
        ]
        rec = pywt.waverec(denoised, wavelet)
        out[:, col] = rec[: len(sig)]
    out = out.astype(x.dtype)
    return out[:, 0] if squeeze else out


def select_active_subcarriers(x: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Drop null/guard subcarrier slots that are ~zero across the session."""
    energy = x.std(axis=0)
    active = energy > threshold
    if not active.any():
        return x
    return x[:, active]


def preprocess(x: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Apply the configured denoising chain to an amplitude matrix.

    Order: null-subcarrier removal -> Hampel -> wavelet -> PCA.
    Returns a new array; input is never mutated.
    """
    out = np.asarray(x, dtype=np.float32).copy()
    if config.drop_null_subcarriers:
        out = select_active_subcarriers(out)
    if config.hampel.enabled:
        out = hampel_filter(
            out, window_size=config.hampel.window_size,
            n_sigmas=config.hampel.n_sigmas,
        )
    if config.wavelet.enabled:
        out = wavelet_denoise(
            out, wavelet=config.wavelet.wavelet,
            level=config.wavelet.level, mode=config.wavelet.mode,
        )
    if config.pca.enabled:
        out = pca_denoise(out, n_components=config.pca.n_components)
    return out
