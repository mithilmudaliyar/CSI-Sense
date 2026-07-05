"""Serial reader for live CSI capture from an ESP32 running ESP32-CSI-Tool.

Design: the reader accepts any object exposing ``readline() -> bytes``
so unit tests (and the dashboard replay mode) can inject a mock instead
of a real serial port. Real hardware use:

    from pipeline.serial_reader import CSISerialReader
    reader = CSISerialReader.open_port("COM5", baudrate=921600)
    for frame in reader.frames():
        ...
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Protocol

from pipeline.parser import CSIFrame, CSIParseError, is_csi_line, parse_csi_line

logger = logging.getLogger(__name__)

DEFAULT_BAUDRATE = 921600  # matches ESP32-CSI-Tool recommended UART config


class LineSource(Protocol):
    """Anything that yields raw bytes lines (pyserial Serial, mock, file)."""

    def readline(self) -> bytes: ...


class MockSerial:
    """Test double: replays a list of pre-baked lines like a serial port."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = [
            line if line.endswith("\n") else line + "\n" for line in lines
        ]
        self._idx = 0

    def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""  # emulate read timeout / end of stream
        line = self._lines[self._idx]
        self._idx += 1
        return line.encode("utf-8", errors="replace")


class CSISerialReader:
    """Reads CSI_DATA lines from a line source and yields parsed frames."""

    def __init__(self, source: LineSource, host_timestamps: bool = True) -> None:
        self._source = source
        self._host_timestamps = host_timestamps
        self.n_frames = 0
        self.n_malformed = 0

    @classmethod
    def open_port(
        cls, port: str, baudrate: int = DEFAULT_BAUDRATE, timeout: float = 1.0
    ) -> "CSISerialReader":
        """Open a real serial port (requires pyserial + hardware)."""
        import serial  # imported lazily so tests never need a device

        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        logger.info("Opened %s @ %d baud", port, baudrate)
        return cls(ser)

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[float, CSIFrame]]:
        """Yield (host_epoch_seconds, CSIFrame) tuples until the source dries up.

        The host timestamp is attached at read time because the ESP32 clock
        is not synced to wall time (see ESP32-CSI-Tool README).
        """
        while True:
            raw = self._source.readline()
            if not raw:
                return  # timeout / end of mock data
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:  # pragma: no cover - decode with replace can't fail
                continue
            if not is_csi_line(line):
                continue
            try:
                frame = parse_csi_line(line)
            except CSIParseError as exc:
                self.n_malformed += 1
                logger.debug("Malformed line skipped: %s", exc)
                continue
            self.n_frames += 1
            host_ts = time.time() if self._host_timestamps else float(self.n_frames)
            yield host_ts, frame
            if max_frames is not None and self.n_frames >= max_frames:
                return
