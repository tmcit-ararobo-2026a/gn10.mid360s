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
        
        self.device = "cuda:0"  # デバイスを指定（例: "cuda:0"）

        # Create/Sensors メニューに「Mid-360S LiDAR」を追加
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

    def _create_sensor(self) -> None:
        """USD Stage 上に Mid-360S センサーを生成"""
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()

        # ステージが開かれていない場合は作成しない
        if not stage:
            print("[Mid360S] Error: Cannot create sensor because no USD stage is open.")
            return

        # パスの重複を防ぐため、空いている Prim パスを取得 (/World/Sensors/Mid360S, /World/Sensors/Mid360S_01, ...)
        base_prim_path = "/World/Sensors/Mid360S"
        sensor_prim_path = omni.usd.get_stage_next_free_path(stage, base_prim_path, False)

        self._warp_manager = WarpUSDManager(stage, device=self.device)
        # LidarSensor の構築
        mesh_ids_array = wp.array([self._warp_manager.wp_mesh.id], dtype=wp.uint64, device=self.device)
        sensor_pos = torch.tensor([[1.0, 0.0, 0.5]], device=self.device)
        sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=self.device)

        env_cfg = {
            "num_envs": 1,
            "mesh_ids": mesh_ids_array,
            "sensor_pos_tensor": sensor_pos,
            "sensor_quat_tensor": sensor_quat,
        }

        # Mid-360S センサーの設定
        lidar_config = LidarConfig(
            sensor_type=LidarType.MID360,
            max_range=30.0,
            enable_sensor_noise=False
        )

        # Mid-360S センサーを生成
        self._sensor = LidarSensor(
            env=env_cfg,
            env_cfg={},
            sensor_config=lidar_config,
            num_sensors=1,
            device=self.device
        )

        print(f"[Mid360S] Sensor created at {sensor_prim_path}")

    def on_shutdown(self) -> None:
        """Shutdown the Mid-360S extension and clean up resources."""

        # メニュー項目の削除
        if hasattr(self, "_menu_items") and self._menu_items:
            omni.kit.menu.utils.remove_menu_items(self._menu_items, "Create/Sensors")
            self._menu_items = []

        gc.collect()

        print("[Mid360S] Extension shutdown.")

    def _on_timeline_play(self, event: object) -> None:
        """Timeline play event callback.

        Args:
            event: The timeline play event.
        """
        print("[Mid360S] Timeline play event received.")
        if not self._physics_subscription:
            self._physics_subscription = self._physics_simulation_interface.subscribe_physics_on_step_events(
                pre_step=False, order=0, on_update=self._on_physics_step
            )

    def _on_timeline_stop(self, event: object) -> None:
        """Timeline stop event callback.

        Args:
            event: The timeline stop event.
        """
        print("[Mid360S] Timeline stop event received.")
        self._physics_subscription = None

    def _on_physics_step(self, step: object, context: object) -> None:
        """Physics step event callback.

        Args:
            step: The physics step event.
            context: The physics context.
        """
        if self._warp_manager:
            self._warp_manager.update_transforms()
        else:
            print("[Mid360S] Warning: Warp manager is not initialized.")
            return

        if self._sensor:
            points_tensor, _ = self._sensor.update()
        else:
            print("[Mid360S] Warning: Lidar sensor is not initialized.")
            return

        pts = points_tensor.detach().cpu().numpy().reshape(-1, 3)

        valid_mask = ~np.isnan(pts).any(axis=1) & (np.linalg.norm(pts, axis=1) > 0.01)
        pts_valid = pts[valid_mask]

        if pts_valid.size > 0:
            print(f"[Mid360S] Valid points count: {pts_valid.shape[0]}")
