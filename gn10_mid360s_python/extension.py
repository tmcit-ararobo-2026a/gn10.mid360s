# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim extension for the Livox Mid-360S LiDAR."""

from __future__ import annotations

import gc

import omni.ext
import omni.usd

from .sensor import Mid360RaycastSensor, Mid360SensorConfig


class Extension(omni.ext.IExt):
    """Isaac Sim Mid-360S LiDAR extension."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize the Mid-360S extension."""

        self._ext_id = ext_id
        self._sensor: Mid360RaycastSensor | None = None

        # Stageイベントを監視するためのサブスクライバ
        usd_context = omni.usd.get_context()
        self._stage_event_sub = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="Mid360S Stage Event"
        )

        # すでにステージが開かれている場合（拡張機能のホットリロード時など）のフォールバック
        if usd_context.get_stage():
            self._create_sensor()

    def _on_stage_event(self, event: omni.usd.StageEvent) -> None:
        """USD Stage のイベントハンドラ"""
        # 新しいステージが開かれた（または作成された）タイミングでセンサーを生成
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._create_sensor()

    def _create_sensor(self) -> None:
        """USD Stage 上に Mid-360S センサーを生成"""
        if self._sensor is not None:
            return

        self._sensor = Mid360RaycastSensor(
            Mid360SensorConfig(
                sensor_prim_path="/World/Sensors/Mid360S",
                translation=(0.0, 0.0, 0.0),
                min_range_m=0.2,
                max_range_m=200.0,
                scan_period_s=0.1,
                points_per_second=200_000,
                output_frame="SENSOR",
            )
        )

        self._sensor.create()

        print(
            "[Mid360S] Sensor created at "
            f"{self._sensor.sensor_prim_path}"
        )

    def on_shutdown(self) -> None:
        """Shutdown the Mid-360S extension."""

        self._stage_event_sub = None
        self._sensor = None

        gc.collect()

        print("[Mid360S] Extension shutdown.")