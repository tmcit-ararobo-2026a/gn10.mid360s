"""ROS2 PointCloud2 publishing utilities for the Mid-360 sensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class PointCloud2PublisherConfig:
    """Configuration for a PointCloud2 ROS2 publisher."""

    topic_name: str = "/mid360/points"
    debug_topic_name: str = "/mid360/debug_points"
    frame_id: str = "mid360"
    node_name: str = "gn10_mid360s_pointcloud_publisher"
    queue_size: int = 10
    max_debug_points: int = 5000
    debug_publish_period: int = 30


class Mid360Ros2Bridge:
    """Minimal ROS2 bridge that publishes Mid-360 point clouds."""

    def __init__(self, config: PointCloud2PublisherConfig | None = None) -> None:
        self._config = config or PointCloud2PublisherConfig()
        self._rclpy: Any = None
        self._node: Any = None
        self._publisher: Any = None
        self._debug_publisher: Any = None
        self._pointcloud2_type: Any = None
        self._pointfield_type: Any = None
        self._header_type: Any = None
        self._owns_context = False
        self._ready = False
        self._debug_publish_count = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> bool:
        """Initialize ROS2 and create the publisher if possible."""
        if self._ready:
            return True

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import PointCloud2, PointField
            from std_msgs.msg import Header
        except ImportError:
            return False

        self._rclpy = rclpy
        self._pointcloud2_type = PointCloud2
        self._pointfield_type = PointField
        self._header_type = Header

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_context = True

        self._node = Node(self._config.node_name)
        qos_profile = QoSProfile(depth=self._config.queue_size)
        qos_profile.reliability = ReliabilityPolicy.RELIABLE
        self._publisher = self._node.create_publisher(PointCloud2, self._config.topic_name, qos_profile)
        self._debug_publisher = self._node.create_publisher(PointCloud2, self._config.debug_topic_name, qos_profile)
        self._ready = True
        return True

    def shutdown(self) -> None:
        """Destroy the ROS2 node and release the context if we own it."""
        if not self._ready:
            return

        try:
            if self._node is not None:
                self._node.destroy_node()
        finally:
            self._node = None
            self._publisher = None
            self._debug_publisher = None
            self._ready = False
            self._debug_publish_count = 0
            if self._owns_context and self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
            self._owns_context = False

    def publish_points(self, points: Sequence[Sequence[float]] | np.ndarray, frame_id: str | None = None) -> bool:
        """Publish a point cloud from Nx3 or Nx4 points.

        The fourth channel is treated as intensity when present; otherwise it is filled with 1.0.
        """
        if not self._is_context_ready():
            return False

        points_array = np.asarray(points, dtype=np.float32)
        if points_array.size == 0:
            cloud = np.zeros(0, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)])
        else:
            if points_array.ndim != 2 or points_array.shape[1] not in (3, 4):
                raise ValueError(f"Expected Nx3 or Nx4 points, got {points_array.shape}")

            if points_array.shape[1] == 3:
                intensity = np.ones((points_array.shape[0], 1), dtype=np.float32)
                points_array = np.concatenate((points_array, intensity), axis=1)

            cloud = np.empty(
                points_array.shape[0],
                dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)],
            )
            cloud["x"] = points_array[:, 0]
            cloud["y"] = points_array[:, 1]
            cloud["z"] = points_array[:, 2]
            cloud["intensity"] = points_array[:, 3]

        try:
            message = self._pointcloud2_type()
            message.header = self._header_type()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = frame_id or self._config.frame_id
            message.height = 1
            message.width = int(cloud.shape[0])
            message.is_bigendian = False
            message.is_dense = bool(cloud.shape[0])
            message.fields = [
                self._make_field("x", 0),
                self._make_field("y", 4),
                self._make_field("z", 8),
                self._make_field("intensity", 12),
            ]
            message.point_step = 16
            message.row_step = message.point_step * message.width
            message.data = cloud.tobytes()

            self._publisher.publish(message)
            return True
        except Exception as error:
            print(f"[Mid360Ros2Bridge] publish_points skipped: {error}")
            return False

    def publish_from_reading(
        self,
        reading: object,
        max_range_m: float,
        frame_id: str | None = None,
    ) -> bool:
        """Convert a raycast sensor reading into PointCloud2 and publish it."""
        if not self._is_context_ready() or reading is None:
            return False

        if not getattr(reading, "is_valid", False):
            return False

        depths = np.asarray(getattr(reading, "depths", []), dtype=np.float32)
        hit_positions = np.asarray(getattr(reading, "hit_positions", []), dtype=np.float32)
        if depths.size == 0 or hit_positions.size == 0:
            return self.publish_points(np.zeros((0, 4), dtype=np.float32), frame_id=frame_id)

        if hit_positions.ndim != 2 or hit_positions.shape[1] < 3:
            return False

        valid_mask = np.isfinite(depths) & (depths < float(max_range_m) - 1e-5)
        valid_points = hit_positions[valid_mask, :3]
        self._log_scan_summary(depths, hit_positions, valid_points, max_range_m=max_range_m)
        self.publish_debug_points(hit_positions[:, :3], valid_mask, frame_id=frame_id)
        return self.publish_points(valid_points, frame_id=frame_id)

    def summarize_reading(self, reading: object, max_range_m: float) -> dict[str, float | int]:
        """Return a compact diagnostic summary for scan debugging."""
        if reading is None:
            return {"valid": 0, "finite_depths": 0, "hits": 0, "points": 0, "min_depth": 0.0, "max_depth": 0.0}

        depths = np.asarray(getattr(reading, "depths", []), dtype=np.float32)
        hit_positions = np.asarray(getattr(reading, "hit_positions", []), dtype=np.float32)
        finite_mask = np.isfinite(depths)
        hit_mask = finite_mask & (depths < float(max_range_m) - 1e-5)
        finite_depths = depths[finite_mask]
        min_depth = float(np.min(finite_depths)) if finite_depths.size else 0.0
        max_depth = float(np.max(finite_depths)) if finite_depths.size else 0.0
        return {
            "valid": int(bool(getattr(reading, "is_valid", False))),
            "finite_depths": int(np.count_nonzero(finite_mask)),
            "hits": int(np.count_nonzero(hit_mask)),
            "points": int(hit_positions.shape[0]) if hit_positions.ndim == 2 else 0,
            "min_depth": min_depth,
            "max_depth": max_depth,
        }

    def publish_debug_points(
        self,
        raw_points: np.ndarray,
        valid_mask: np.ndarray,
        frame_id: str | None = None,
    ) -> bool:
        """Publish a debug cloud that keeps miss rays visible in RViz2."""
        if not self._is_context_ready() or self._debug_publisher is None:
            return False

        self._debug_publish_count += 1
        if self._config.debug_publish_period > 1 and (self._debug_publish_count % self._config.debug_publish_period) != 0:
            return False

        points_array = np.asarray(raw_points, dtype=np.float32)
        if points_array.ndim != 2 or points_array.shape[1] < 3:
            return False

        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape[0] != points_array.shape[0]:
            return False

        point_count = points_array.shape[0]
        max_points = max(1, int(self._config.max_debug_points))
        if point_count > max_points:
            sample_indices = np.linspace(0, point_count - 1, max_points, dtype=np.int64)
            points_array = points_array[sample_indices]
            valid_mask = valid_mask[sample_indices]

        cloud = np.empty(
            points_array.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)],
        )
        cloud["x"] = points_array[:, 0]
        cloud["y"] = points_array[:, 1]
        cloud["z"] = points_array[:, 2]
        cloud["intensity"] = np.where(valid_mask, 1.0, 0.1).astype(np.float32)

        try:
            message = self._pointcloud2_type()
            message.header = self._header_type()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = frame_id or self._config.frame_id
            message.height = 1
            message.width = int(cloud.shape[0])
            message.is_bigendian = False
            message.is_dense = False
            message.fields = [
                self._make_field("x", 0),
                self._make_field("y", 4),
                self._make_field("z", 8),
                self._make_field("intensity", 12),
            ]
            message.point_step = 16
            message.row_step = message.point_step * message.width
            message.data = cloud.tobytes()

            self._debug_publisher.publish(message)
            return True
        except Exception as error:
            print(f"[Mid360Ros2Bridge] publish_debug_points skipped: {error}")
            return False

    def _log_scan_summary(
        self,
        depths: np.ndarray,
        hit_positions: np.ndarray,
        valid_points: np.ndarray,
        *,
        max_range_m: float,
    ) -> None:
        if self._node is None:
            return

        valid_count = int(valid_points.shape[0])
        finite_count = int(np.count_nonzero(np.isfinite(depths)))
        hit_count = int(np.count_nonzero(np.isfinite(depths) & (depths < float(max_range_m) - 1e-5)))
        finite_depths = depths[np.isfinite(depths)]
        min_depth = float(np.min(finite_depths)) if finite_depths.size else 0.0
        max_depth = float(np.max(finite_depths)) if finite_depths.size else 0.0

        if valid_count == 0:
            print(
                "[Mid360Ros2Bridge] scan produced no valid points "
                f"(finite_depths={finite_count}, hits={hit_count}, raw_points={hit_positions.shape[0]}, "
                f"min_depth={min_depth:.4f}, max_depth={max_depth:.4f})"
            )
            return

        first_point = valid_points[0].tolist()
        print(
            "[Mid360Ros2Bridge] scan summary: "
            f"valid_points={valid_count}, finite_depths={finite_count}, hits={hit_count}, "
            f"min_depth={min_depth:.4f}, max_depth={max_depth:.4f}, first_point={first_point}"
        )

    def _is_context_ready(self) -> bool:
        if not self._ready or self._node is None or self._rclpy is None:
            return False

        try:
            if not self._rclpy.ok():
                return False
        except Exception:
            return False

        node_context = getattr(self._node, "context", None)
        if node_context is not None:
            try:
                if not node_context.ok():
                    return False
            except Exception:
                return False

        return True

    def _make_field(self, name: str, offset: int) -> Any:
        return self._pointfield_type(name=name, offset=offset, datatype=self._pointfield_type.FLOAT32, count=1)
