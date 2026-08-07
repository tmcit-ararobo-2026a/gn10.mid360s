"""Load Livox Mid-360 scan pattern data from a NumPy file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mid360Pattern:
    """Container for raycast sensor pattern arrays."""

    ray_origins: np.ndarray
    ray_directions: np.ndarray
    ray_time_offsets: np.ndarray


def get_default_mid360_pattern_path() -> Path:
    """Return the default path of the Mid-360 scan pattern."""
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "patterns"
        / "mid360.npy"
    )


def _angles_to_directions(
    azimuth: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """Convert azimuth/elevation angles to normalized XYZ directions."""

    cos_el = np.cos(elevation)

    directions = np.column_stack(
        (
            cos_el * np.cos(azimuth),
            cos_el * np.sin(azimuth),
            np.sin(elevation),
        )
    )

    norms = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    # Numerical safety.
    norms[norms == 0.0] = 1.0

    return directions / norms


def load_mid360_pattern(
    pattern_path: str | Path | None = None,
    scan_period_s: float = 0.1,
) -> Mid360Pattern:
    """Load Mid-360 scan pattern from a NumPy file.

    Expected format:

        Nx2:
            [azimuth_rad, elevation_rad]

        Nx3:
            [azimuth_rad, elevation_rad, time_offset]

    The returned ray origins and directions are expressed
    in the sensor-local coordinate system.
    """

    if scan_period_s <= 0.0:
        raise ValueError(
            f"scan_period_s must be > 0, got {scan_period_s}"
        )

    path = (
        Path(pattern_path)
        if pattern_path is not None
        else get_default_mid360_pattern_path()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Mid-360 pattern file not found: {path}"
        )

    samples = np.load(path)

    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(
            f"Unsupported Mid-360 pattern shape: {samples.shape}; "
            "expected Nx2 or Nx3"
        )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    azimuth = samples[:, 0]
    elevation = samples[:, 1]

    ray_directions = _angles_to_directions(
        azimuth,
        elevation,
    ).astype(
        np.float32,
        copy=False,
    )

    # Every ray starts at the sensor origin.
    ray_origins = np.zeros(
        (samples.shape[0], 3),
        dtype=np.float32,
    )

    if samples.shape[1] >= 3:
        raw_offsets = samples[:, 2]

        raw_min = float(np.min(raw_offsets))
        raw_max = float(np.max(raw_offsets))

        if (
            raw_min >= 0.0
            and raw_max <= scan_period_s
        ):
            ray_time_offsets = raw_offsets.astype(
                np.float32,
                copy=False,
            )
        else:
            # Normalize arbitrary timing information into
            # the configured scan period.
            span = max(
                raw_max - raw_min,
                1e-8,
            )

            normalized = (
                (raw_offsets - raw_min)
                / span
            )

            ray_time_offsets = (
                normalized * scan_period_s
            ).astype(
                np.float32,
                copy=False,
            )

    else:
        # No timing information exists in the file.
        # Distribute the rays uniformly across one scan period.
        ray_count = samples.shape[0]

        ray_time_offsets = (
            np.arange(
                ray_count,
                dtype=np.float32,
            )
            / float(ray_count)
            * scan_period_s
        )

    return Mid360Pattern(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        ray_time_offsets=ray_time_offsets,
    )