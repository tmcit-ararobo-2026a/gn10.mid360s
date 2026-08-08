import warp as wp
from pxr import Usd, UsdGeom
import numpy as np
import omni.timeline

class WarpUSDManager:
    """Utility class to manage Warp mesh representation of USD Meshes in the stage."""

    def __init__(self, stage: Usd.Stage, device: str = "cuda:0"):
        self.stage = stage
        self.device = device

        self.mesh_prims = []
        self.wp_pts = None
        self.wp_idx = None
        self.wp_mesh = None

        self._build_warp_mesh()

    def _build_warp_mesh(self):
        all_vertices = []
        all_indices = []
        index_offset = 0

        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh_geom = UsdGeom.Mesh(prim)
            points = mesh_geom.GetPointsAttr().Get()
            face_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
            face_counts = mesh_geom.GetFaceVertexCountsAttr().Get()

            if points is None or face_indices is None:
                continue

            pts_np = np.array(points, dtype=np.float32)
            indices_np = np.array(face_indices, dtype=np.int32)

            if len(pts_np) == 0 or len(indices_np) == 0:
                continue

            tri_indices = []
            curr_idx = 0
            for count in face_counts:
                for i in range(1, count - 1):
                    tri_indices.append(indices_np[curr_idx])
                    tri_indices.append(indices_np[curr_idx + i])
                    tri_indices.append(indices_np[curr_idx + i + 1])
                curr_idx += count

            tri_indices_np = np.array(tri_indices, dtype=np.int32)

            transform = mesh_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform_matrix = np.array(transform, dtype=np.float32)

            pts_homo = np.hstack([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)])
            pts_transformed = (pts_homo @ transform_matrix)[:, :3]

            all_vertices.append(pts_transformed)
            all_indices.append(tri_indices_np + index_offset)
            index_offset += len(pts_transformed)

            self.mesh_prims.append((mesh_geom, pts_homo))

        if not all_vertices:
            raise RuntimeError("USD stage has no valid mesh geometries.")

        merged_vertices = np.vstack(all_vertices).astype(np.float32)
        merged_indices = np.concatenate(all_indices).astype(np.int32)

        self.wp_pts = wp.array(merged_vertices, dtype=wp.vec3, device=self.device)
        self.wp_idx = wp.array(merged_indices, dtype=int, device=self.device)

        self.wp_mesh = wp.Mesh(points=self.wp_pts, indices=self.wp_idx)

    def update_transforms(self):
        all_transformed_pts = []

        timeline = omni.timeline.get_timeline_interface()
        current_time_sec = timeline.get_current_time()
        fps = self.stage.GetTimeCodesPerSecond()
        time_code = Usd.TimeCode(current_time_sec * fps)

        for mesh_geom, cached_pts_homo in self.mesh_prims:
            points = mesh_geom.GetPointsAttr().Get(time_code)
            if points is not None and len(points) > 0:
                pts_np = np.array(points, dtype=np.float32)
                pts_homo = np.hstack([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)])
            else:
                pts_homo = cached_pts_homo

            transform = mesh_geom.ComputeLocalToWorldTransform(time_code)
            transform_matrix = np.array(transform, dtype=np.float32)

            pts_transformed = (pts_homo @ transform_matrix)[:, :3]
            all_transformed_pts.append(pts_transformed)

        if not all_transformed_pts:
            return

        updated_vertices = np.vstack(all_transformed_pts).astype(np.float32)

        wp.copy(self.wp_pts, wp.array(updated_vertices, dtype=wp.vec3, device=self.device))
        self.wp_mesh.refit()