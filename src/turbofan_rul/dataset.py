"""Read complete engine histories and the separate official test targets."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import stat
from zipfile import BadZipFile, ZipFile

import numpy as np


NASA_ARCHIVE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
NASA_ARCHIVE_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
ARCHIVE_LIMIT = 16 * 1024 * 1024
TEXT_LIMIT = 8 * 1024 * 1024
SENSOR_COUNT = 21
ENGINE_COUNT = 100
# FD001 fitting-engine extrema from run 46001071161, with one 0.0001 input step of tolerance.
FD001_SETTINGS_MIN = (-0.0088, -0.0007, 99.9999)
FD001_SETTINGS_MAX = (0.0088, 0.0007, 100.0001)


@dataclass(frozen=True)
class Trajectory:
    unit: int
    sensors: np.ndarray
    operating_settings: np.ndarray

    @property
    def cycles(self) -> int:
        return len(self.sensors)


@dataclass(frozen=True)
class FD001:
    training: tuple[Trajectory, ...]
    test: tuple[Trajectory, ...]
    test_rul: np.ndarray
    archive_sha256: str


def supports_fd001_conditions(engine: Trajectory) -> bool:
    settings = engine.operating_settings
    return settings.shape == (engine.cycles, 3) and bool(
        np.all((settings >= FD001_SETTINGS_MIN) & (settings <= FD001_SETTINGS_MAX))
    )


def read_bounded(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
            raise ValueError(f"{path} must be a regular file containing 1 to {limit} bytes")
        content = stream.read(metadata.st_size + 1)
    if len(content) != metadata.st_size:
        raise ValueError(f"{path} changed size while being read")
    return content


def verify_archive(content: bytes) -> str:
    digest = sha256(content).hexdigest()
    if digest != NASA_ARCHIVE_SHA256:
        raise ValueError("NASA C-MAPSS archive SHA-256 does not match the verified revision")
    return digest


def _member(archive: ZipFile, name: str, limit: int) -> bytes:
    matches = [entry for entry in archive.infolist() if entry.filename == name]
    if len(matches) != 1 or not 0 < matches[0].file_size <= limit:
        raise ValueError(f"expected exactly one bounded archive member named {name}")
    with archive.open(matches[0]) as stream:
        content = stream.read(limit + 1)
    if len(content) != matches[0].file_size:
        raise ValueError(f"invalid size for archive member {name}")
    return content


def _matrix(content: bytes, name: str, columns: int) -> np.ndarray:
    if not content.strip():
        raise ValueError(f"{name} is empty")
    try:
        values = np.loadtxt(BytesIO(content), dtype=np.float64, comments=None, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid numeric row") from exc
    if values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"{name} must have {columns} finite columns")
    if np.max(np.abs(values)) > 1e9:
        raise ValueError(f"{name} contains values outside the supported range")
    return values


def parse_trajectories(content: bytes, name: str) -> tuple[Trajectory, ...]:
    rows = _matrix(content, name, columns=5 + SENSOR_COUNT)
    keys = rows[:, :2]
    if (keys < 1).any() or not np.array_equal(keys, np.floor(keys)):
        raise ValueError(f"{name} must have positive integer unit IDs and cycles")

    changes = np.flatnonzero(rows[1:, 0] != rows[:-1, 0]) + 1
    histories = []
    last_unit = 0
    for block in np.split(rows, changes):
        unit = int(block[0, 0])
        if unit <= last_unit or not np.array_equal(block[:, 1], np.arange(1, len(block) + 1)):
            raise ValueError(f"{name} must contain ascending units and consecutive cycles from one")
        histories.append(Trajectory(unit, block[:, 5:].copy(), block[:, 2:5].copy()))
        last_unit = unit
    return tuple(histories)


def load_fd001(path: Path) -> FD001:
    content = read_bounded(path, ARCHIVE_LIMIT)
    digest = verify_archive(content)
    outer_member = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
    try:
        with ZipFile(BytesIO(content)) as outer:
            inner_content = _member(outer, outer_member, ARCHIVE_LIMIT)
        with ZipFile(BytesIO(inner_content)) as archive:
            training = parse_trajectories(
                _member(archive, "train_FD001.txt", TEXT_LIMIT), "train_FD001.txt"
            )
            test = parse_trajectories(
                _member(archive, "test_FD001.txt", TEXT_LIMIT), "test_FD001.txt"
            )
            targets = _matrix(_member(archive, "RUL_FD001.txt", TEXT_LIMIT), "RUL_FD001.txt", 1)
    except (BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("expected the NASA C-MAPSS ZIP bundle") from exc

    units = tuple(range(1, ENGINE_COUNT + 1))
    if any(tuple(engine.unit for engine in split) != units for split in (training, test)):
        raise ValueError("FD001 must contain engines 1 to 100 in each split")
    if (len(targets) != ENGINE_COUNT or (targets < 0).any()
            or not np.array_equal(targets, np.floor(targets))):
        raise ValueError("expected 100 non-negative integer FD001 test targets")
    return FD001(training, test, targets.ravel(), digest)
