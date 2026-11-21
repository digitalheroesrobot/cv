#!/usr/bin/env python3
"""Online RGB-D SLAM front-end and sparse keyframe/pose-graph back-end."""
from dataclasses import dataclass
import numpy as np

from common import DepthMap
from common.geometry import invert_se3
from scene_reconstruction.geometry_pipeline import depth_to_point_cloud
from visual_odometry import ORBRGBDVisualOdometry, rectify_rgbd, estimate_rgbd_pnp
from sensor_fusion import RGBDVisualICPFusion
from .keyframe_manager import KeyframeManager, LoopCandidateDetector
from .pose_graph import PoseGraph, PoseGraphEdge
from .geometric_loop_closure import RGBDLoopClosureVerifier, rebuild_global_map


@dataclass
class SLAMFrameResult:
    success: bool
    frame_index: int
    timestamp: float
    T_world_camera: np.ndarray
    keyframe_added: bool
    loop_closures_added: int
    optimized: bool
    map_points: int
    message: str


class RealTimeRGBDSLAM:
    """Integrate rectified RGB-D frames into a keyframe pose graph.

    The first camera frame defines world origin. VO estimates T_current_previous;
    ICP estimates T_previous_current. The fusion front-end maintains the
    current pose. Keyframe graph edges use T_i_j = inv(T_world_i) @ T_world_j.
    Only loop candidates that pass RGB-D PnP, ICP convergence, RMSE, and
    PnP/ICP consistency gates become graph constraints.
    """
    def __init__(self, K, min_depth_m=.20, max_depth_m=5.0,
                 cloud_stride=5, keyframe_translation_m=.20,
                 keyframe_rotation_rad=.15, keyframe_frame_gap=15,
                 loop_min_id_gap=10, loop_min_matches=25,
                 loop_min_inliers=15, loop_max_rmse_m=.035,
                 loop_max_translation_gap_m=.12,
                 loop_max_rotation_gap_deg=7.0,
                 registration_voxel_m=.04, map_voxel_m=.035,
                 max_map_points=250000, distortion=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.distortion = (np.zeros(5, dtype=np.float64) if distortion is None
                           else np.asarray(distortion, dtype=np.float64).reshape(-1))
        self.loop_min_inliers = int(loop_min_inliers)
        if self.K.shape != (3, 3):
            raise ValueError("K must be a 3x3 camera matrix")
        self.min_depth_m, self.max_depth_m = float(min_depth_m), float(max_depth_m)
        self.cloud_stride = int(cloud_stride)
        if self.cloud_stride < 1 or self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("Invalid depth range or cloud stride")
        self.vo = ORBRGBDVisualOdometry(
            self.K, min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
            max_features=2500, min_matches=20, min_pnp_inliers=12)
        self.fusion = RGBDVisualICPFusion(
            registration_voxel_m=registration_voxel_m,
            map_voxel_m=map_voxel_m, max_map_points=max_map_points)
        self.keyframes = KeyframeManager(
            keyframe_translation_m, keyframe_rotation_rad, keyframe_frame_gap)
        self.loop_detector = LoopCandidateDetector(
            min_id_gap=loop_min_id_gap, min_good_matches=loop_min_matches)
        self.loop_verifier = RGBDLoopClosureVerifier(
            voxel_size_m=registration_voxel_m,
            max_correspondence_m=.15, min_correspondences=30,
            max_rmse_m=loop_max_rmse_m,
            max_translation_gap_m=loop_max_translation_gap_m,
            max_rotation_gap_rad=np.deg2rad(loop_max_rotation_gap_deg))
        self.graph = PoseGraph()
        self.frame_index = 0
        self._orb = self._create_orb()
        self._added_loop_pairs = set()
        self.global_map = np.empty((0, 3), dtype=np.float64)
        self.last_optimized = False

    @staticmethod
    def _create_orb():
        import cv2
        return cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8,
                              edgeThreshold=19, fastThreshold=12)

    def _extract_keyframe_features(self, image):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    @staticmethod
    def _match_keyframes(candidate, current, ratio=.75):
        import cv2
        if candidate.descriptors is None or current.descriptors is None:
            return []
        if len(candidate.descriptors) < 2 or len(current.descriptors) < 2:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(candidate.descriptors, current.descriptors, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < ratio * n.distance]
        return good

    def _try_loop_closures(self, current_keyframe):
        added = 0
        candidates = [kf for kf in self.keyframes.keyframes[:-1]
                      if current_keyframe.keyframe_id - kf.keyframe_id >= self.loop_detector.min_id_gap]
        # Descriptor ranking is a proposal stage, never an acceptance decision.
        ranked = self.loop_detector.detect(current_keyframe, candidates)
        for _, candidate in ranked[:5]:
            pair = (candidate.keyframe_id, current_keyframe.keyframe_id)
            if pair in self._added_loop_pairs:
                continue
            matches = self._match_keyframes(candidate, current_keyframe)
            if len(matches) < self.loop_detector.min_good_matches:
                continue
            prev_uv = np.asarray([candidate.keypoints[m.queryIdx].pt for m in matches],
                                 dtype=np.float64)
            curr_uv = np.asarray([current_keyframe.keypoints[m.trainIdx].pt for m in matches],
                                 dtype=np.float64)
            try:
                T_current_candidate, inliers, pnp_rmse, _ = estimate_rgbd_pnp(
                    prev_uv, curr_uv, candidate.depth_m, self.K,
                    min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
                    min_inliers=self.loop_min_inliers, reprojection_threshold_px=2.5)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            if inliers < self.loop_min_inliers or not np.isfinite(pnp_rmse) or pnp_rmse > 3.0:
                continue
            verification = self.loop_verifier.verify_cloud_pair(
                candidate.point_cloud, current_keyframe.point_cloud,
                T_current_candidate)
            if not verification.accepted:
                continue
            # The verified measurement maps current-camera coordinates into
            # candidate-camera coordinates: exactly the graph's T_i_j convention.
            self.graph.add_edge(PoseGraphEdge(
                candidate.keyframe_id, current_keyframe.keyframe_id,
                verification.T_candidate_current,
                translation_information=8.0,
                rotation_information=8.0,
                kind="verified_rgbd_loop"))
            self._added_loop_pairs.add(pair)
            added += 1
        return added

    def _optimize_and_rebuild(self):
        result = self.graph.optimize(max_nfev=80, loss="huber", f_scale=.05)
        for keyframe, pose in zip(self.keyframes.keyframes, result.poses):
            keyframe.T_world_camera = pose.copy()
        self.global_map = rebuild_global_map(
            self.keyframes.keyframes, result.poses,
            voxel_size_m=self.fusion.map_voxel_m,
            max_points=self.fusion.max_map_points)
        if self.keyframes.keyframes:
            # Re-anchor the sequential front-end at the corrected latest
            # keyframe pose. The previous local cloud remains in camera coords.
            self.fusion.T_world_previous = result.poses[-1].copy()
            self.fusion.map_points_world = self.global_map.copy()
        self.last_optimized = True
        return result

    def process(self, image_bgr, depth_m, timestamp):
        image, depth = rectify_rgbd(
            image_bgr, depth_m, self.K, self.distortion)
        depth_map = DepthMap(depth, float(timestamp), "rgbd_slam", unit="m")
        cloud = depth_to_point_cloud(
            depth_map, self._calibration(), stride=self.cloud_stride,
            min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m)
        points = cloud.points
        if len(points) < 30:
            raise RuntimeError("Too few valid depth points for SLAM registration")

        visual = self.vo.process(image, depth)
        if self.fusion.previous_cloud is None:
            self.fusion.initialize(points)
            T_world_camera = np.eye(4)
            fusion_success = True
            fusion_message = "initialized"
        else:
            fused = self.fusion.integrate(points, visual)
            fusion_success = fused.success
            T_world_camera = fused.T_world_current.copy()
            fusion_message = fused.message
            if not fused.success:
                # VO has consumed this image, while the fusion state rejected
                # its cloud. Restart a clean local submap so the next VO pair
                # and the ICP previous cloud refer to the same camera frame.
                self.reset()
                visual = self.vo.process(image, depth)
                self.fusion.initialize(points)
                T_world_camera = np.eye(4)
                fusion_success = True
                fusion_message = (
                    "tracking lost; restarted local submap at current frame")

        keypoints, descriptors = self._extract_keyframe_features(image)
        tracking_quality = (
            visual.pnp_inliers / max(visual.feature_matches, 1)
            if visual.success else (0.5 if fusion_success else 0.0))
        should_add = self.keyframes.should_insert(
            self.frame_index, T_world_camera, tracking_quality)
        added_loops, optimized = 0, False
        if should_add:
            keyframe = self.keyframes.add(
                self.frame_index, timestamp, T_world_camera,
                image=image.copy(), depth_m=depth.copy(),
                descriptors=descriptors, keypoints=keypoints,
                point_cloud=points.copy())
            node_id = self.graph.add_node(T_world_camera)
            if node_id > 0:
                previous_keyframe_pose = self.graph.poses[node_id - 1]
                T_previous_keyframe_current = (
                    invert_se3(previous_keyframe_pose) @ T_world_camera)
                self.graph.add_edge(PoseGraphEdge(
                    node_id - 1, node_id, T_previous_keyframe_current,
                    translation_information=1.0,
                    rotation_information=1.0,
                    kind="keyframe_odometry"))
            added_loops = self._try_loop_closures(keyframe)
            if added_loops:
                self._optimize_and_rebuild()
                T_world_camera = self.fusion.T_world_previous.copy()
                optimized = True
            else:
                self.global_map = rebuild_global_map(
                    self.keyframes.keyframes, self.graph.poses,
                    voxel_size_m=self.fusion.map_voxel_m,
                    max_points=self.fusion.max_map_points)
        else:
            self.global_map = self.fusion.map_points_world.copy()

        message = fusion_message
        if not visual.success:
            message += "; visual odometry unavailable"
        if added_loops:
            message += f"; accepted {added_loops} verified loop closure(s)"
        self.frame_index += 1
        return SLAMFrameResult(
            True, self.frame_index - 1, float(timestamp), T_world_camera.copy(),
            bool(should_add), int(added_loops), bool(optimized),
            len(self.global_map), message)

    def _calibration(self):
        # Depth is already rectified; distortion is zero and only K is used
        # by the depth back-projection function.
        from common import CameraCalibration
        h, w = self._image_shape
        return CameraCalibration(self.K, w, h)

    def set_image_shape(self, image_shape):
        self._image_shape = tuple(image_shape[:2])

    def reset(self):
        self.vo.reset()
        self.fusion.reset()
        self.keyframes.keyframes.clear()
        self.keyframes.last_selected_frame = -1
        self.graph = PoseGraph()
        self.frame_index = 0
        self._added_loop_pairs.clear()
        self.global_map = np.empty((0, 3), dtype=np.float64)
