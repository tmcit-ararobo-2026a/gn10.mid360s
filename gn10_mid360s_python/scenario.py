# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scenario implementation for the scripting workflow template."""

from __future__ import annotations

import numpy as np
from typing import Any
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.utils.viewports import set_camera_view

from .mid360_sensor import Mid360RaycastSensor, Mid360SensorConfig
from .ros2_bridge import Mid360Ros2Bridge, PointCloud2PublisherConfig


class FrankaRmpFlowExampleScript:
    """Script a Franka robot to move to a target using RMPFlow."""

    def __init__(self) -> None:
        self._mid360_sensor: Any | None = None
        self._mid360_sensor_builder: Mid360RaycastSensor = Mid360RaycastSensor(
            Mid360SensorConfig(
                sensor_prim_path="/World/Sensors/Mid360S",
                translation=(0.0, 0.0, 1.0),
                min_range_m=0.2,
                max_range_m=200.0,
                scan_period_s=0.1,
                output_frame="SENSOR",
            )
        )
        self._mid360_ros2_bridge: Mid360Ros2Bridge = Mid360Ros2Bridge(
            PointCloud2PublisherConfig(
                topic_name="/mid360/points",
                debug_topic_name="/mid360/debug_points",
                frame_id="mid360",
                node_name="gn10_mid360s_pointcloud_publisher",
                queue_size=10,
            )
        )

        self._mid360_debug_step = 0
        self._debug_scene_objects = []

    def load_example_assets(self) -> tuple:
        """Create a minimal sensor debug scene and return loaded assets."""
        self._debug_scene_objects = [
            FixedCuboid(
                name="wall",
                prim_path="/World/debug_wall",
                scale=np.array([0.2, 3.0, 2.5]),
                position=np.array([2.0, 0.0, 1.0]),
                color=np.array([1.0, 0.4, 0.0]),
            ),
            FixedCuboid(
                name="ob1",
                prim_path="/World/obstacle_1",
                scale=np.array([0.03, 1.0, 0.3]),
                position=np.array([0.25, 0.25, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
            FixedCuboid(
                name="ob2",
                prim_path="/World/obstacle_2",
                scale=np.array([0.5, 0.03, 0.3]),
                position=np.array([0.5, 0.25, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
            FixedCuboid(
                name="ob3",
                prim_path="/World/obstacle_3",
                scale=np.array([0.2, 0.2, 0.6]),
                position=np.array([1.0, 0.0, 0.3]),
                color=np.array([0.0, 1.0, 0.0]),
            ),
            GroundPlane("/World/Ground"),
        ]

        return tuple(self._debug_scene_objects)

    def setup(self) -> None:
        """Initialize the sensor-only scene."""
        set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        self._mid360_sensor = self._mid360_sensor_builder.create()
        if not self._mid360_ros2_bridge.initialize():
            print("ROS2 is not available; Mid-360 PointCloud2 publishing is disabled.")

    def reset(self) -> None:
        """Reset sensor debug counters."""
        self._mid360_debug_step = 0

    def cleanup(self) -> None:
        """Release ROS2 resources owned by this scenario."""
        self._mid360_ros2_bridge.shutdown()

    def update_ros2(self) -> None:
        """Publish the current Mid-360 point cloud if the sensor is active."""
        if self._mid360_sensor is None:
            return

        sensor = self._mid360_sensor
        reading = sensor.get_sensor_reading()
        self._mid360_debug_step += 1
        summary = self._mid360_ros2_bridge.summarize_reading(reading, max_range_m=self._mid360_sensor_builder.get_max_range_m())
        if self._mid360_debug_step % 60 == 0 or summary["hits"] == 0:
            print(
                "[Mid360] reading summary: "
                f"valid={summary['valid']}, finite_depths={summary['finite_depths']}, "
                f"hits={summary['hits']}, points={summary['points']}, "
                f"min_depth={summary['min_depth']:.4f}, max_depth={summary['max_depth']:.4f}"
            )
        self._mid360_ros2_bridge.publish_from_reading(
            reading,
            max_range_m=self._mid360_sensor_builder.get_max_range_m(),
            frame_id=self._mid360_sensor_builder.get_frame_id(),
        )
