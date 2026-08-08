# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim extension for the Livox Mid-360S LiDAR."""

from __future__ import annotations

import gc

import carb.eventdispatcher
import omni.ext
import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription
import omni.usd
import omni.physics.core
import omni.timeline
from pxr import Usd, UsdGeom
import torch
import warp as wp
import numpy as np

# for Mid-360
from .OmniPerception.LidarSensor.LidarSensor.lidar_sensor import LidarSensor
from .OmniPerception.LidarSensor.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
# for Warp Mesh
from .warp_manager import WarpUSDManager


class Extension(omni.ext.IExt):
    """Isaac Sim Mid-360S LiDAR extension."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize the Mid-360S extension and register menu items."""

        self._ext_id = ext_id

        self._warp_manager = None
        self._sensor = None
        self._sensor_prim = None
        self._stage = None
        
        self.device = "cuda:0"

        # Menu items
        self._menu_items = [
            MenuItemDescription(
                name="Mid-360S LiDAR",
                onclick_fn=lambda: self._create_sensor(),
            )
        ]
        omni.kit.menu.utils.add_menu_items(self._menu_items, "Create/Sensors")

        # Physics
        self._physics_simulation_interface = (
            omni.physics.core.get_physics_simulation_interface()
        )
        self._physics_subscription = None

        # Timeline
        self._timeline = omni.timeline.get_timeline_interface()

        self._timeline_event_sub_play = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_PLAY,
            on_event=self._on_timeline_play,
            observer_name="template_extension._on_timeline_play",
        )
        self._timeline_event_sub_stop = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=self._on_timeline_stop,
            observer_name="template_extension._on_timeline_stop",
        )

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

        if not self._stage:
            print("[Mid360S] Error: Cannot create sensor because no USD stage is open.")
            return

        if self._sensor is not None:
            print("[Mid360S] Sensor already exists!!")
            return

        # Create a new Xform prim for the sensor in the USD stage
        base_prim_path = "/World/Sensors/Mid360S"
        sensor_prim_path = omni.usd.get_stage_next_free_path(self._stage, base_prim_path, False)

        self._sensor_prim = self._stage.DefinePrim(sensor_prim_path, "Xform")

        print(f"[Mid360S] Sensor prim created at {sensor_prim_path}")

    def _update_sensor_pose(self) -> None:
        """Read the sensor pose from the USD stage."""

        if self._sensor_prim is None:
            return

        if self._sensor is None:
            return

        xformable = UsdGeom.Xformable(
            self._sensor_prim
        )

        world_transform = xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()

        imag = rotation.GetImaginary()
        real = rotation.GetReal()

        position = torch.tensor(
            [
                [
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        quaternion = torch.tensor(
            [
                [
                    [
                        float(imag[0]),
                        float(imag[1]),
                        float(imag[2]),
                        float(real),
                    ]
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self._sensor.lidar_positions_tensor.copy_(
            position
        )

        self._sensor.lidar_quat_tensor.copy_(
            quaternion
        )

    def _on_timeline_play(self, event: object) -> None:
        """Timeline play event callback.

        Args:
            event: The timeline play event.
        """
        print("[Mid360S] Timeline play event received.")
        if self._stage is None:
            usd_context = omni.usd.get_context()
            self._stage = usd_context.get_stage()
        if self._sensor_prim is None:
            self._sensor_prim = self._stage.GetPrimAtPath("/World/Sensors/Mid360S")

        # Physics step subscription
        if not self._physics_subscription:
            self._physics_subscription = self._physics_simulation_interface.subscribe_physics_on_step_events(
                pre_step=False, order=0, on_update=self._on_physics_step
            )
        
        if self._warp_manager is not None:
            print("[Mid360S] WarpUSDManager already exists!!")
            return

        # WarpUSDManager
        self._warp_manager = WarpUSDManager(self._stage, device=self.device)

        # LidarSensor
        mesh_ids_array = wp.array([self._warp_manager.wp_mesh.id], dtype=wp.uint64, device=self.device)
        sensor_pos = torch.tensor([[1.0, 0.0, 0.5]], device=self.device)
        sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=self.device)

        env_cfg = {
            "num_envs": 1,
            "mesh_ids": mesh_ids_array,
            "sensor_pos_tensor": sensor_pos,
            "sensor_quat_tensor": sensor_quat,
        }

        if self._sensor is not None:
            print("[Mid360S] Sensor already exists!!")
            return

        # Mid-360S configuration
        lidar_config = LidarConfig(
            sensor_type=LidarType.MID360,
            max_range=30.0,
            enable_sensor_noise=False
        )

        # Mid-360S sensor creation
        self._sensor = LidarSensor(
            env=env_cfg,
            env_cfg={},
            sensor_config=lidar_config,
            num_sensors=1,
            device=self.device
        )

        print(f"[Mid360S] Sensor created with config: {lidar_config}")

    def _on_timeline_stop(self, event: object) -> None:
        """Timeline stop event callback.

        Args:
            event: The timeline stop event.
        """
        print("[Mid360S] Timeline stop event received.")
        self._physics_subscription = None
        self._warp_manager = None
        self._sensor = None

    def _on_physics_step(self, step: object, context: object) -> None:
        """Physics step event callback.

        Args:
            step: The physics step event.
            context: The physics context.
        """
        if self._sensor is None:
            print("[Mid360S] No sensor available.")
            return
        if self._warp_manager is None:
            print("[Mid360S] No warp manager available.")
            return
        if self._sensor_prim is None:
            print("[Mid360S] No sensor prim available.")
            return

        self._update_sensor_pose()
        self._warp_manager.update_transforms()
        points_tensor, _ = self._sensor.update()

        if points_tensor is None:
            print("[Mid360S] No points generated by the sensor.")
            return

        pts = points_tensor.detach().cpu().numpy().reshape(-1, 3)

        valid_mask = ~np.isnan(pts).any(axis=1) & (np.linalg.norm(pts, axis=1) > 0.01)
        pts_valid = pts[valid_mask]

        if pts_valid.size <= 0:
            print("[Mid360S] No valid points generated by the sensor.")
            return
