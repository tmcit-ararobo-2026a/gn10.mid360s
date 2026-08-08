import omni.graph.core as og
import omni.timeline
import numpy as np
import torch
import warp as wp
from pxr import Usd, UsdGeom

from ..OmniPerception.LidarSensor.LidarSensor.lidar_sensor import LidarSensor
from ..OmniPerception.LidarSensor.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
from ..impl.warp_manager import WarpUSDManager


class OgnMid360PointCloud:
    """Mid-360S LiDARの生成・更新・点群出力を行うノード。"""

    class InternalState:
        def __init__(self):
            self.stage = None
            self.sensor_prim = None
            self.warp_manager = None
            self.sensor = None
            self.device = "cuda:0"
            self.timeline_sub_play = None
            self.timeline_sub_stop = None
            self.physics_sub = None
            self.latest_points = None

        def is_ready(self) -> bool:
            return self.sensor is not None and self.warp_manager is not None

    @staticmethod
    def internal_state():
        return OgnMid360PointCloud.InternalState()

    @staticmethod
    def initialize(context, node):
        """ノードがグラフに追加された時に1回だけ呼ばれる。Timelineの購読をここで張る。"""
        state: OgnMid360PointCloud.InternalState = node.get_node_type().get_internal_state(node) \
            if hasattr(node.get_node_type(), "get_internal_state") else None
        # 実際のKitバージョンによりinternal_stateの取り出し方が異なるため、
        # 代わりにcompute側の遅延初期化(下記)に寄せるのが安全です。

    @staticmethod
    def release(node):
        """ノードがグラフから削除された時に1回だけ呼ばれる。購読解除・GPUリソース解放。"""
        pass

    @staticmethod
    def compute(db) -> bool:
        state = db.internal_state

        # --- 遅延初期化: Play開始後の最初のcomputeで、指定Primからセンサーを組み立てる ---
        if not state.is_ready():
            sensor_targets = db.inputs.sensorPrim
            if not sensor_targets:
                db.outputs.points = []
                return True  # まだPrimが割り当てられていない

            sensor_path = str(sensor_targets[0].GetPrimPath())  # 型は要検証(下記注記)
            stage = omni.usd.get_context().get_stage()
            state.stage = stage
            state.sensor_prim = stage.GetPrimAtPath(sensor_path)

            state.warp_manager = WarpUSDManager(
                stage, device=state.device
            )

            mesh_ids_array = wp.array([state.warp_manager.wp_mesh.id], dtype=wp.uint64, device=state.device)
            sensor_pos = torch.zeros((1, 3), device=state.device)
            sensor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=state.device)

            lidar_config = LidarConfig(sensor_type=LidarType.MID360, max_range=30.0, enable_sensor_noise=False)
            state.sensor = LidarSensor(
                env={"num_envs": 1, "mesh_ids": mesh_ids_array,
                     "sensor_pos_tensor": sensor_pos, "sensor_quat_tensor": sensor_quat},
                env_cfg={}, sensor_config=lidar_config, num_sensors=1, device=state.device,
            )

        # --- 毎フレーム: Poseを読んでレイキャスト ---
        OgnMid360PointCloud._update_sensor_pose(state)
        state.warp_manager.update_transforms()
        points_tensor, _ = state.sensor.update()

        if points_tensor is None:
            db.outputs.points = []
            return True

        points = points_tensor.detach().cpu().numpy().reshape(-1, 3)
        valid_mask = ~np.isnan(points).any(axis=1) & (np.linalg.norm(points, axis=1) > 0.01)
        points = points[valid_mask]

        db.outputs.points = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def _update_sensor_pose(state) -> None:
        xformable = UsdGeom.Xformable(state.sensor_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()
        imag, real = rotation.GetImaginary(), rotation.GetReal()

        state.sensor.lidar_positions_tensor.copy_(
            torch.tensor([[float(translation[0]), float(translation[1]), float(translation[2])]],
                         dtype=torch.float32, device=state.device)
        )
        state.sensor.lidar_quat_tensor.copy_(
            torch.tensor([[[float(imag[0]), float(imag[1]), float(imag[2]), float(real)]]],
                         dtype=torch.float32, device=state.device)
        )