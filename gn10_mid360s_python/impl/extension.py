# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Gento Aiba.
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

"""Isaac Sim extension for the Livox Mid-360S LiDAR."""

from __future__ import annotations

import gc

import omni.graph.core as og
import omni.ext
import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription
import omni.usd


class Extension(omni.ext.IExt):
    """Isaac Sim Mid-360S LiDAR extension."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize the Mid-360S extension and register menu items."""

        self._ext_id = ext_id

        # Menu items
        self._menu_items = [
            MenuItemDescription(
                name="Mid-360S LiDAR",
                onclick_fn=lambda: self._create_sensor(),
            )
        ]
        omni.kit.menu.utils.add_menu_items(self._menu_items, "Create/Sensors")

    def on_shutdown(self) -> None:
        """Shutdown the Mid-360S extension and clean up resources."""

        # Unsubscribe from timeline events
        if hasattr(self, "_menu_items") and self._menu_items:
            omni.kit.menu.utils.remove_menu_items(self._menu_items, "Create/Sensors")
            self._menu_items = []
        gc.collect()

        print("[Mid360S] Extension shutdown.")

    def _create_sensor(self) -> None:
        """Create a Mid-360S LiDAR sensor in the USD stage."""
        usd_context = omni.usd.get_context()
        self._stage = usd_context.get_stage()

        # Create a new Xform prim for the sensor in the USD stage
        base_prim_path = "/World/Sensors/Mid360S"
        sensor_prim_path = omni.usd.get_stage_next_free_path(self._stage, base_prim_path, False)

        self._sensor_prim = self._stage.DefinePrim(sensor_prim_path, "Xform")

        print(f"[Mid360S] Sensor prim created at {sensor_prim_path}")

        graph_path = f"{sensor_prim_path}/ActionGraph"

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacSimlationGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                    ("Mid360PointCloud", "gn10.mid360s.Mid360PointCloud"),
                    ("ROS2PublishPointCloud", "isaacsim.ros2.bridge.ROS2PublishPointCloud"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "IsaacSimlationGate.inputs:execIn"),
                    ("OnPlaybackTick.outputs:time", "ROS2PublishPointCloud.inputs:timeStamp"),
                    ("IsaacSimlationGate.outputs:execOut", "Mid360PointCloud.inputs:execIn"),
                    ("Mid360PointCloud.outputs:points", "ROS2PublishPointCloud.inputs:data"),
                    ("Mid360PointCloud.outputs:execOut", "ROS2PublishPointCloud.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("IsaacSimlationGate.inputs:step", 2),
                    ("Mid360PointCloud.inputs:sensorPrim", [sensor_prim_path]),
                    ("ROS2PublishPointCloud.inputs:topicName", "/livox/lidar"),
                    ("ROS2PublishPointCloud.inputs:frameId", "livox_frame"),
                ],
            },
        )