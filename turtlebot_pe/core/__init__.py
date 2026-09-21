"""
Core algorithmic modules for pursuit-evasion.
Pure Python and NumPy implementations with ZERO ROS2 dependencies.
"""

from .math_utils import wrap_angle, euclidean_distance, unit_vector, clamp
from .ekf_tracker import TargetEKF
from .lidar_detector import LidarTargetDetector
from .obstacle_map import ObstacleMap
from .catcher_controller import CatcherController

__all__ = [
    "wrap_angle",
    "euclidean_distance",
    "unit_vector",
    "clamp",
    "TargetEKF",
    "LidarTargetDetector",
    "ObstacleMap",
    "CatcherController",
]
