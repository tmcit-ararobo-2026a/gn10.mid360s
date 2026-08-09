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

"""Mid360 Lidar Point Cloud Node for OmniGraph."""

import numpy as np
import torch
import warp as wp

import omni.graph.core as og
import omni.usd
from pxr import Usd, UsdGeom

from ..OmniPerception.LidarSensor.LidarSensor.lidar_sensor import LidarSensor
from ..OmniPerception.LidarSensor.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
from ..impl.warp_manager import WarpUSDManager


class OgnMid360PointCloud:
    """Mid360 Lidar Point Cloud Node for OmniGraph."""

    class InternalState:
        def __init__(self):
            self.stage = None
            self.sensor_prim = None
            self.warp_manager = None
            self.sensor = None
            self.device = "cuda:0" # CUDA device for Warp

        def is_ready(self) -> bool:
            """Check if the internal state is ready for point cloud computation."""
            return self.sensor is not None and self.warp_manager is not None

    @staticmethod
    def internal_state():
        return OgnMid360PointCloud.InternalState()

    @staticmethod
    def initialize(context, node):
        state: OgnMid360PointCloud.InternalState = node.get_node_type().get_internal_state(node) \
            if hasattr(node.get_node_type(), "get_internal_state") else None

    @staticmethod
    def release(node):
        pass

    @staticmethod
    def compute(db) -> bool:
        """Scans the environment and outputs the point cloud data."""
        state = db.internal_state

        if not state.is_ready():
            sensor_targets = db.inputs.sensorPrim
            if not sensor_targets:
                db.outputs.points = []
                return True

            sensor_path = str(sensor_targets[0].GetPrimPath())
            stage = omni.usd.get_context().get_stage()
            state.stage = stage
            state.sensor_prim = stage.GetPrimAtPath(sensor_path)

            state.warp_manager = WarpUSDManager(
                stage, device=state.device
            )

            # Initialize the LidarSensor with the appropriate configuration
            mesh_ids_array = wp.array([state.warp_manager.wp_mesh.id], dtype=wp.uint64, device=state.device)
            sensor_pos = torch.zeros((1, 3), device=state.device)
            sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=state.device)

            lidar_config = LidarConfig(sensor_type=LidarType.MID360, max_range=30.0, enable_sensor_noise=False)
            state.sensor = LidarSensor(
                env={"num_envs": 1, "mesh_ids": mesh_ids_array,
                     "sensor_pos_tensor": sensor_pos, "sensor_quat_tensor": sensor_quat},
                env_cfg={}, sensor_config=lidar_config, num_sensors=1, device=state.device,
            )

        # Update the sensor pose and compute the point cloud
        OgnMid360PointCloud._update_sensor_pose(state)
        state.warp_manager.update_transforms()
        points_tensor, _ = state.sensor.update()

        if points_tensor is None:
            db.outputs.points = []
            return True
        
        # GPU to CPU transfer and filtering of invalid points
        points = points_tensor.detach().cpu().numpy().reshape(-1, 3)
        valid_mask = ~np.isnan(points).any(axis=1) & (np.linalg.norm(points, axis=1) > 0.01)
        points = points[valid_mask]

        # Convert points to a list of tuples for output
        db.outputs.points = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED # Enable execution output
        return True

    @staticmethod
    def _update_sensor_pose(state) -> None:
        """Update the sensor's position and orientation based on the USD stage."""
        xformable = UsdGeom.Xformable(state.sensor_prim) # Get the Xformable interface for the sensor prim
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()
        imag, real = rotation.GetImaginary(), rotation.GetReal()

        # Update the sensor's position and orientation tensors in the internal state
        state.sensor.lidar_positions_tensor.copy_(
            torch.tensor([[float(translation[0]), float(translation[1]), float(translation[2])]],
                         dtype=torch.float32, device=state.device)
        )
        state.sensor.lidar_quat_tensor.copy_(
            torch.tensor([[[float(imag[0]), float(imag[1]), float(imag[2]), float(real)]]],
                         dtype=torch.float32, device=state.device)
        )