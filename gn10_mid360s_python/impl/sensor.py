# SPDX-License-Identifier: Apache-2.0

"""Livox Mid-360S Physics Raycast Sensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import omni.usd
from pxr import UsdGeom

from .pattern_loader import Mid360Pattern, load_mid360_pattern


@dataclass(frozen=True)
class Mid360SensorConfig:
    """Configuration for the Mid-360S Physics Raycast Sensor."""

    sensor_prim_path: str = "/World/Sensors/Mid360S"

    translation: Sequence[float] = (0.0, 0.0, 0.0)

    min_range_m: float = 0.2
    max_range_m: float = 200.0

    scan_period_s: float = 0.1

    # Livox Mid-360S maximum point rate.
    points_per_second: int = 200_000

    output_frame: str = "SENSOR"

    report_hit_prim_paths: bool = False


class Mid360RaycastSensor:
    """Author and manage a Livox Mid-360S Physics Raycast Sensor."""

    def __init__(
        self,
        config: Mid360SensorConfig | None = None,
        pattern_path: str | None = None,
    ) -> None:
        self._config = config or Mid360SensorConfig()
        self._pattern_path = pattern_path

        self._pattern: Mid360Pattern | None = None
        self._sensor = None
        self._authored_sensor = None

    @property
    def sensor(self):
        """Return the runtime RaycastSensor wrapper."""
        return self._sensor

    @property
    def sensor_prim_path(self) -> str:
        """Return the USD path of the sensor."""
        return self._config.sensor_prim_path

    @property
    def ray_count(self) -> int:
        """Return the number of rays in one simulation step."""
        if self._pattern is None:
            return 0

        return int(self._pattern.ray_directions.shape[0])

    @property
    def points_per_second(self) -> int:
        """Return the configured point rate."""
        return self._config.points_per_second

    def create(self) -> "Mid360RaycastSensor":
        """Create the USD Prim and initialize the Physics Raycast backend."""

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError(
                "No USD stage is open. "
                "Open a stage before creating the Mid-360S sensor."
            )

        self._ensure_parent_xform(stage)

        # Load the Livox scan pattern.
        self._pattern = load_mid360_pattern(
            None,
            scan_period_s=self._config.scan_period_s,
        )

        from isaacsim.sensors.experimental.physics import (
            Raycast,
            RaycastSensor,
        )

        # Create the Physics Raycast Sensor USD prim.
        self._authored_sensor = Raycast.create(
            self._config.sensor_prim_path,
            translations=[
                list(self._config.translation)
            ],
            min_range=self._config.min_range_m,
            max_range=self._config.max_range_m,
            ray_origins=self._pattern.ray_origins,
            ray_directions=self._pattern.ray_directions,
            ray_time_offsets=self._pattern.ray_time_offsets,
            output_frame=self._config.output_frame,
            report_hit_prim_paths=self._config.report_hit_prim_paths,
        )

        # Create the runtime sensor wrapper.
        self._sensor = RaycastSensor(self._authored_sensor)

        print(
            "[Mid360S] Physics Raycast Sensor initialized:\n"
            f"  Prim: {self._config.sensor_prim_path}\n"
            f"  Rays/scan: {self.ray_count}\n"
            f"  Points/sec: {self._config.points_per_second}\n"
            f"  Scan period: {self._config.scan_period_s}s\n"
            f"  Range: "
            f"{self._config.min_range_m} - "
            f"{self._config.max_range_m} m"
        )

        return self

    def get_sensor_reading(self):
        """Return the latest sensor reading."""

        if self._sensor is None:
            return None

        return self._sensor.get_sensor_reading()

    def _ensure_parent_xform(self, stage) -> None:
        """Create the parent Xform if it does not already exist."""

        parent_path = self._config.sensor_prim_path.rsplit("/", 1)[0]

        if not parent_path:
            parent_path = "/World"

        parent_prim = stage.GetPrimAtPath(parent_path)

        if parent_prim.IsValid():
            return

        UsdGeom.Xform.Define(stage, parent_path)