"""Session storage: append CSI frames to per-session files with labels.

Layout (one session = one labeled recording):

    data/raw/<session_id>.csv           # wide table: metadata + amp_0..amp_N + phase_0..phase_N
    data/raw/<session_id>.meta.json     # sidecar: label, node id, notes, schema info

Session ids are timestamped: 20260706-143000_walking_node1.
Parquet is used instead of CSV when pyarrow is importable and
``prefer_parquet=True`` (CSV stays the default so captures are always
inspectable and diffable).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.parser import CSIFrame

logger = logging.getLogger(__name__)

VALID_LABELS = ("empty", "walking", "fall", "2people", "unlabeled")


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    label: str
    node_id: str
    started_utc: str
    sample_rate_hz_nominal: float
    n_subcarriers: int
    notes: str = ""
    source: str = "unknown"  # "serial" | "synthetic" | "replay"


def make_session_id(label: str, node_id: str, when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"{when.strftime('%Y%m%d-%H%M%S')}_{label}_{node_id}"


def frames_to_dataframe(stamped_frames: list[tuple[float, CSIFrame]]) -> pd.DataFrame:
    """Convert (host_ts, CSIFrame) tuples into a tidy wide DataFrame."""
    if not stamped_frames:
        raise ValueError("No frames to convert")
    n_sc = stamped_frames[0][1].n_subcarriers
    rows = []
    for host_ts, f in stamped_frames:
        if f.n_subcarriers != n_sc:
            logger.warning(
                "Dropping frame with %d subcarriers (expected %d)",
                f.n_subcarriers, n_sc,
            )
            continue
        row = {
            "host_ts": host_ts,
            "local_timestamp_us": f.local_timestamp_us,
            "rssi": f.rssi,
            "mac": f.mac,
            "role": f.role,
        }
        amp = f.amplitude
        ph = f.phase
        for i in range(n_sc):
            row[f"amp_{i}"] = float(amp[i])
        for i in range(n_sc):
            row[f"phase_{i}"] = float(ph[i])
        rows.append(row)
    return pd.DataFrame(rows)


def amplitude_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the (n_samples, n_subcarriers) amplitude matrix from a session df."""
    amp_cols = sorted(
        (c for c in df.columns if c.startswith("amp_")),
        key=lambda c: int(c.split("_")[1]),
    )
    if not amp_cols:
        raise ValueError("DataFrame has no amp_* columns")
    return df[amp_cols].to_numpy(dtype=np.float32)


class SessionWriter:
    """Appends CSI frames to a session file with a metadata sidecar."""

    def __init__(
        self,
        out_dir: str | Path,
        label: str,
        node_id: str = "node1",
        notes: str = "",
        source: str = "serial",
        sample_rate_hz_nominal: float = 50.0,
        prefer_parquet: bool = False,
    ) -> None:
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = make_session_id(label, node_id)
        self.label = label
        self.node_id = node_id
        self.notes = notes
        self.source = source
        self.sample_rate_hz_nominal = sample_rate_hz_nominal
        self._use_parquet = prefer_parquet and _parquet_available()
        self._buffer: list[tuple[float, CSIFrame]] = []
        self._n_written = 0
        self._n_subcarriers: int | None = None

    @property
    def data_path(self) -> Path:
        ext = "parquet" if self._use_parquet else "csv"
        return self.out_dir / f"{self.session_id}.{ext}"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.session_id}.meta.json"

    def append(self, host_ts: float, frame: CSIFrame) -> None:
        if self._n_subcarriers is None:
            self._n_subcarriers = frame.n_subcarriers
        self._buffer.append((host_ts, frame))
        if len(self._buffer) >= 256:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        df = frames_to_dataframe(self._buffer)
        if self._use_parquet:
            # Parquet append: read+concat is fine at hobby-project scale.
            if self.data_path.exists():
                df = pd.concat([pd.read_parquet(self.data_path), df], ignore_index=True)
            df.to_parquet(self.data_path, index=False)
        else:
            header = not self.data_path.exists()
            df.to_csv(self.data_path, mode="a", header=header, index=False)
        self._n_written += len(self._buffer)
        self._buffer = []
        self._write_meta()

    def close(self) -> SessionMeta:
        self.flush()
        meta = self._meta()
        self._write_meta()
        logger.info(
            "Session %s closed: %d frames -> %s",
            self.session_id, self._n_written, self.data_path,
        )
        return meta

    def _meta(self) -> SessionMeta:
        return SessionMeta(
            session_id=self.session_id,
            label=self.label,
            node_id=self.node_id,
            started_utc=datetime.now(timezone.utc).isoformat(),
            sample_rate_hz_nominal=self.sample_rate_hz_nominal,
            n_subcarriers=self._n_subcarriers or 0,
            notes=self.notes,
            source=self.source,
        )

    def _write_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(asdict(self._meta()), indent=2), encoding="utf-8"
        )


def load_session(data_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load a session file plus its sidecar metadata."""
    data_path = Path(data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    meta_path = data_path.with_suffix("").with_suffix(".meta.json") \
        if data_path.suffix else data_path
    meta_path = data_path.parent / f"{data_path.stem}.meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        logger.warning("No metadata sidecar for %s", data_path.name)
    return df, meta


def list_sessions(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        p for p in data_dir.iterdir()
        if p.suffix in (".csv", ".parquet") and not p.name.endswith(".meta.json")
    )


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False
