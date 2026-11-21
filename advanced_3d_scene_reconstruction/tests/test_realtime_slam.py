#!/usr/bin/env python3
"""Offline tests for Lesson 49 integrated SLAM artifacts."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from keyframe_slam import PoseGraph, PoseGraphEdge, Keyframe, rebuild_global_map
from lesson49_realtime_rgbd_slam import save_ply


def make_pose(x=0., y=0., yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.]
    return T


class TestRealtimeSLAMArtifacts(unittest.TestCase):
    def test_pose_graph_and_rebuild_pipeline(self):
        world = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.]])
        poses = [make_pose(), make_pose(1., 0.)]
        local = [world, world - np.array([1., 0., 0.])]
        frames = [Keyframe(i, float(i), poses[i], point_cloud=local[i])
                  for i in range(2)]
        graph = PoseGraph()
        for T in poses:
            graph.add_node(T)
        graph.add_edge(PoseGraphEdge(0, 1, np.linalg.inv(poses[0]) @ poses[1]))
        result = graph.optimize()
        cloud = rebuild_global_map(frames, result.poses, voxel_size_m=.001)
        self.assertEqual(len(cloud), 3)
        self.assertLess(result.final_cost, 1e-10)

    def test_ply_writer_creates_valid_header_and_vertices(self):
        points = np.array([[0., 1., 2.], [3., 4., 5.]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.ply"
            save_ply(path, points)
            text = path.read_text(encoding="ascii")
            self.assertIn("element vertex 2", text)
            self.assertIn("end_header", text)
            self.assertIn("3.000000 4.000000 5.000000", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
