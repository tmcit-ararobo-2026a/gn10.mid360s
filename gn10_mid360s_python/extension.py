# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim extension for the Livox Mid-360S LiDAR."""

from __future__ import annotations

import gc

import omni.ext
import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription
import omni.usd

from .sensor import Mid360RaycastSensor, Mid360SensorConfig


class Extension(omni.ext.IExt):
    """Isaac Sim Mid-360S LiDAR extension."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize the Mid-360S extension and register menu items."""

        self._ext_id = ext_id
        self._sensors: list[Mid360RaycastSensor] = []

        # Create/Sensors メニューに「Mid-360S LiDAR」を追加
        self._menu_items = [
            MenuItemDescription(
                name="Mid-360S LiDAR",
                onclick_fn=lambda: self._create_sensor(),
            )
        ]
        omni.kit.menu.utils.add_menu_items(self._menu_items, "Create/Sensors")

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

        sensor = Mid360RaycastSensor(
            Mid360SensorConfig(
                sensor_prim_path=sensor_prim_path,
                translation=(0.0, 0.0, 0.0),
                min_range_m=0.2,
                max_range_m=200.0,
                scan_period_s=0.1,
                points_per_second=200_000,
                output_frame="SENSOR",
            )
        )

        sensor.create()
        self._sensors.append(sensor)

        print(f"[Mid360S] Sensor created at {sensor_prim_path}")

    def on_shutdown(self) -> None:
        """Shutdown the Mid-360S extension and clean up resources."""

        # メニュー項目の削除
        if hasattr(self, "_menu_items") and self._menu_items:
            omni.kit.menu.utils.remove_menu_items(self._menu_items, "Create/Sensors")
            self._menu_items = []

        self._sensors.clear()

        gc.collect()

        print("[Mid360S] Extension shutdown.")