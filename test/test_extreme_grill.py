import math
import numpy as np
import pytest

from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.obstacle_map import ObstacleMap

def test_catcher_target_much_faster():
    """Test catcher behavior when target is impossibly fast. It should clamp commands safely."""
    ctrl = CatcherController(v_max=0.3, omega_max=1.8)
    ctrl.set_state("PURSUE")
    
    ego_pose = (0.0, 0.0, 0.0)
    # Target moving at 5.0 m/s (way faster than max 0.3)
    target_state = np.array([2.0, 2.0, math.pi/4, 5.0, 0.0])
    
    v, w, state = ctrl.compute_control(ego_pose, (0.3, 0.0), target_state, [], dt=0.05, bg_ready=True)
    
    assert state == "PURSUE"
    assert v <= 0.3
    assert abs(w) <= 1.8
    assert not math.isnan(v)
    assert not math.isnan(w)

def test_lidar_multiple_clusters_ekf_association():
    """Test that when there are multiple valid clusters, detector picks the one closest to EKF prediction."""
    detector = LidarTargetDetector(
        bg_learning_duration=0.1, 
        target_type="any",
        min_cluster_size=1
    )
    
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams
    
    # Train background (walls at 5.0m)
    bg_scan = [5.0] * num_beams
    for step in range(5):
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=step*0.05)
        
    assert detector.is_background_ready()
    
    # Inject 2 targets
    scan = list(bg_scan)
    
    # Target 1: at 2.0m directly ahead (beam 180)
    b1 = int((0.0 - angle_min) / angle_inc)
    scan[b1] = 2.0
    
    # Target 2: at 1.0m to the left (beam 270)
    b2 = int((math.pi/2 - angle_min) / angle_inc)
    scan[b2] = 1.0
    
    # Let's say EKF predicts target is around 2.0m ahead.
    detector.set_ekf_prediction(predicted_range=2.1, predicted_bearing=0.0)
    
    detection = detector.process_scan(scan, angle_min, angle_inc, timestamp=0.5)
    
    # It should pick Target 1 (2.0m ahead) instead of Target 2, even if Target 2 is closer
    assert detection is not None
    assert math.isclose(detection[0], 2.0, abs_tol=0.1)
    assert math.isclose(detection[1], 0.0, abs_tol=0.1)

def test_obstacle_map_bresenham_vertical_horizontal():
    """Test Bresenham raycasting for purely vertical and horizontal lines."""
    omap = ObstacleMap(resolution=0.1, size=10.0, arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    omap.grid[:] = 0 # clear
    omap._is_finalized = True
    
    # Place an obstacle directly in the horizontal path (x=60, y=50)
    omap.grid[60, 50] = 1
    
    # x goes from 0.0 (index 50) to 2.0 (index 70). y is 0.0 (index 50)
    has_los = omap.has_line_of_sight((0.0, 0.0), (2.0, 0.0))
    assert not has_los, "Failed to detect horizontal obstacle"
    
    # Place an obstacle directly in the vertical path (x=50, y=60)
    omap.grid[50, 60] = 1
    has_los = omap.has_line_of_sight((0.0, 0.0), (0.0, 2.0))
    assert not has_los, "Failed to detect vertical obstacle"

def test_ekf_covariance_growth_on_occlusion():
    """Test that EKF covariance grows when handling occlusion."""
    ekf = TargetEKF(dt=0.05)
    ekf.init_state(0.0, 0.0, 0.0, 0.0, 0.0)
    
    P_initial = np.trace(ekf.get_covariance())
    
    # Occlusion happens
    ekf.handle_occlusion()
    
    P_after = np.trace(ekf.get_covariance())
    
    assert P_after > P_initial, "Covariance did not grow on occlusion"
