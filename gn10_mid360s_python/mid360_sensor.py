"""Mid-360 Physics Raycast Sensor creation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import omni.usd
from pxr import UsdGeom

from .pattern_loader import Mid360Pattern, load_mid360_pattern


@dataclass
class Mid360SensorConfig:
    """Configuration for Mid-360 Physics Raycast Sensor."""

    sensor_prim_path: str = "/World/Sensors/Mid360S"
    translation: Sequence[float] = (0.0, 0.0, 1.0)
    min_range_m: float = 0.2
    max_range_m: float = 200.0
    scan_period_s: float = 0.1
    output_frame: str = "WORLD"


class Mid360RaycastSensor:
    """Create and hold a Mid-360 Physics Raycast Sensor instance."""

    def __init__(self, config: Mid360SensorConfig | None = None, pattern_path: str | None = None) -> None:
        self._config = config or Mid360SensorConfig()
        self._pattern_path = pattern_path
        self._pattern: Mid360Pattern | None = None
        self._sensor = None

    @property
    def ray_count(self) -> int:
        if self._pattern is None:
            return 0
        return int(self._pattern.ray_directions.shape[0])

    @property
    def sensor(self) -> object:
        return self._sensor

    def create(self) -> object:
        """Create the USD raycast sensor prim and runtime wrapper."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is open. Cannot create Mid-360 sensor.")

        self._pattern = load_mid360_pattern(self._pattern_path, scan_period_s=self._config.scan_period_s)
        self._ensure_parent_xform(stage)

        from isaacsim.sensors.experimental.physics import Raycast, RaycastSensor

        kwargs = dict(
            min_range=self._config.min_range_m,
            max_range=self._config.max_range_m,
            ray_origins=self._pattern.ray_origins,
            ray_directions=self._pattern.ray_directions,
            ray_time_offsets=self._pattern.ray_time_offsets,
            output_frame=self._config.output_frame,
            translations=[list(self._config.translation)],
        )

        try:
            authored_sensor = Raycast.create(self._config.sensor_prim_path, **kwargs)
        except TypeError:
            # Compatibility fallback for builds that accept only Python lists.
            kwargs["ray_origins"] = self._pattern.ray_origins.tolist()
            kwargs["ray_directions"] = self._pattern.ray_directions.tolist()
            kwargs["ray_time_offsets"] = self._pattern.ray_time_offsets.tolist()
            authored_sensor = Raycast.create(self._config.sensor_prim_path, **kwargs)

        self._sensor = RaycastSensor(authored_sensor)
        return self._sensor

    def _ensure_parent_xform(self, stage: object) -> None:
        parent_path = self._config.sensor_prim_path.rsplit("/", 1)[0]
        if not parent_path:
            parent_path = "/World"
        if stage.GetPrimAtPath(parent_path):
            return
        UsdGeom.Xform.Define(stage, parent_path)
