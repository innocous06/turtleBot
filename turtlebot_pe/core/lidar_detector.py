import math
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from .math_utils import wrap_angle


class LidarTargetDetector:
    def __init__(
        self,
        bg_learning_duration: float = 2.0,
        foreground_threshold: float = 0.15,
        min_cluster_size: int = 1,
        max_cluster_size: int = 30,
        cluster_distance_threshold: float = 0.30,
        min_target_width: float = 0.04,
        max_target_width: float = 0.60,
        target_type: str = "any",
    ):
        self.bg_learning_duration = float(bg_learning_duration)
        self.foreground_threshold = float(foreground_threshold)
        self.min_cluster_size = int(min_cluster_size)
        self.max_cluster_size = int(max_cluster_size)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.min_target_width = float(min_target_width)
        self.max_target_width = float(max_target_width)
        self.target_type = target_type

        if self.target_type == "robot":
            self.min_target_width = max(0.15, self.min_target_width)
            self.min_cluster_size = max(3, self.min_cluster_size)

        self.bg_ranges: Optional[np.ndarray] = None
        self.bg_scan_count = 0
        self.start_timestamp: Optional[float] = None
        self._is_bg_ready = False

        self.ekf_pred_range: Optional[float] = None
        self.ekf_pred_bearing: Optional[float] = None
        self.latest_clusters: List[Dict[str, Any]] = []

    def set_ekf_prediction(self, predicted_range: Optional[float], predicted_bearing: Optional[float]) -> None:
        self.ekf_pred_range = predicted_range
        self.ekf_pred_bearing = predicted_bearing

    def is_background_ready(self) -> bool:
        return self._is_bg_ready

    def get_detected_clusters(self) -> List[Dict[str, Any]]:
        return self.latest_clusters

    def process_scan(
        self,
        ranges: List[float],
        angle_min: float,
        angle_increment: float,
        timestamp: float,
        range_min: float = 0.15,
        range_max: float = 12.0,
    ) -> Optional[Tuple[float, float]]:
        arr_ranges = np.array(ranges, dtype=float)
        arr_ranges = np.where(np.isnan(arr_ranges), range_max, arr_ranges)
        arr_ranges = np.where(np.isinf(arr_ranges), range_max, arr_ranges)

        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        elapsed_learning = timestamp - self.start_timestamp

        # BG learning
        if not self._is_bg_ready:
            if self.bg_ranges is None or len(self.bg_ranges) != len(arr_ranges):
                self.bg_ranges = arr_ranges.copy()
                self.bg_scan_count = 1
            else:
                self.bg_ranges = (self.bg_ranges * self.bg_scan_count + arr_ranges) / (self.bg_scan_count + 1)
                self.bg_scan_count += 1

            if elapsed_learning >= self.bg_learning_duration and self.bg_scan_count >= 5:
                self._is_bg_ready = True
            return None

        # FG extraction
        num_beams = len(arr_ranges)
        if self.bg_ranges is None or len(self.bg_ranges) != num_beams:
            return None

        is_foreground = (
            (arr_ranges < (self.bg_ranges - self.foreground_threshold))
            & (arr_ranges >= range_min)
            & (arr_ranges <= range_max)
        )
        foreground_indices = np.where(is_foreground)[0]
        if len(foreground_indices) == 0:
            self.latest_clusters = []
            return None

        angles = angle_min + foreground_indices * angle_increment
        fg_r = arr_ranges[foreground_indices]
        fg_x = fg_r * np.cos(angles)
        fg_y = fg_r * np.sin(angles)
        fg_points = np.column_stack((fg_x, fg_y, angles, fg_r))

        # Clustering
        clusters = []
        current_cluster = [fg_points[0]]

        for i in range(1, len(fg_points)):
            prev_pt = fg_points[i - 1][:2]
            curr_pt = fg_points[i][:2]
            dist = float(np.linalg.norm(curr_pt - prev_pt))
            if dist <= self.cluster_distance_threshold:
                current_cluster.append(fg_points[i])
            else:
                clusters.append(np.array(current_cluster))
                current_cluster = [fg_points[i]]

        if current_cluster:
            clusters.append(np.array(current_cluster))

        if len(clusters) > 1:
            first_pt = clusters[0][0][:2]
            last_pt = clusters[-1][-1][:2]
            if float(np.linalg.norm(first_pt - last_pt)) <= self.cluster_distance_threshold:
                clusters[0] = np.vstack((clusters[-1], clusters[0]))
                clusters.pop()

        # Filtering
        valid_clusters = []
        self.latest_clusters = []

        for c in clusters:
            count = len(c)
            if not (self.min_cluster_size <= count <= self.max_cluster_size):
                continue

            pts = c[:, :2]
            if count == 1:
                width = 0.05
            else:
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=-1))
                width = float(np.max(dists))

            if not (self.min_target_width <= width <= self.max_target_width):
                continue

            centroid = np.mean(pts, axis=0)
            cx, cy = float(centroid[0]), float(centroid[1])
            c_range = math.sqrt(cx * cx + cy * cy)
            c_bearing = wrap_angle(math.atan2(cy, cx))

            cluster_info = {
                "centroid": (cx, cy),
                "range": c_range,
                "bearing": c_bearing,
                "width": width,
                "count": count,
                "points": pts,
            }
            valid_clusters.append(cluster_info)
            self.latest_clusters.append(cluster_info)

        if not valid_clusters:
            return None

        # Data association
        if len(valid_clusters) == 1:
            best = valid_clusters[0]
            return (best["range"], best["bearing"])

        if self.ekf_pred_range is not None and self.ekf_pred_bearing is not None:
            pred_x = self.ekf_pred_range * math.cos(self.ekf_pred_bearing)
            pred_y = self.ekf_pred_range * math.sin(self.ekf_pred_bearing)
            best_cluster = min(
                valid_clusters,
                key=lambda cl: math.hypot(cl["centroid"][0] - pred_x, cl["centroid"][1] - pred_y),
            )
            return (best_cluster["range"], best_cluster["bearing"])

        best_cluster = min(valid_clusters, key=lambda cl: cl["range"])
        return (best_cluster["range"], best_cluster["bearing"])
