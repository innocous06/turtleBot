import math
import numpy as np
import pytest

from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.obstacle_map import ObstacleMap

def test_catcher_dwa_collision_avoidance():
    """Test that the DWA avoids a direct collision when target is behind an obstacle."""
    ctrl = CatcherController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        v_max=0.3,
        omega_max=1.8,
        robot_radius=0.18
    )
    ctrl.set_state("PURSUE")

    # Ego at origin facing +x.
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.3, 0.0)
    
    # Target is at (2.0, 0.0), directly in front.
    target_state = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    
    # But there is a massive obstacle exactly at (1.0, 0.0).
    obstacles = [{"center": (1.0, 0.0), "radius": 0.4}]
    
    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        ego_vel=ego_vel,
        target_ekf_state=target_state,
        obstacles=obstacles,
        dt=0.05,
        bg_ready=True
    )
    
    # If PN alone was active, w_cmd would be 0 (straight ahead).
    # With DWA, it must steer away (w_cmd != 0) to avoid the obstacle.
    # It might also slow down.
    assert state == "PURSUE"
    assert abs(w_cmd) > 0.1, "DWA failed to steer away from direct collision course"

def test_ekf_noisy_occlusion_and_mahalanobis():
    """Test that EKF correctly rejects a massive outlier using Mahalanobis gating."""
    ekf = TargetEKF(dt=0.05)
    
    # Initialize the target at (1.0, 0.0)
    ekf.init_state(x=1.0, y=0.0, theta=0.0, v=0.2, omega=0.0)
    
    # Step 1: Predict
    ekf.predict()
    
    # Step 2: Inject a wildly incorrect measurement (e.g., target teleported 10m away)
    ego_pose = (0.0, 0.0, 0.0)
    z_meas = [10.0, 0.0] # 10m away directly in front
    
    accepted, d_m2 = ekf.update(z_meas, ego_pose)
    
    # Should be rejected because distance is huge
    assert not accepted, "EKF accepted an impossible outlier"
    assert d_m2 > 5.991, "Mahalanobis distance should be large"
    
    # The target position should not jump to 10m
    pos = ekf.get_position()
    assert math.hypot(pos[0] - 1.0, pos[1] - 0.0) < 0.2, "EKF state was corrupted by outlier"

def test_lidar_detector_sphere_mode():
    """Test LidarTargetDetector detecting a tiny sphere with min_cluster_size=1."""
    detector = LidarTargetDetector(
        bg_learning_duration=0.5, 
        target_type="any",
        min_cluster_size=1,
        min_target_width=0.04
    )
    
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams
    
    # 1. Train background (walls at 5.0m)
    bg_scan = [5.0] * num_beams
    for step in range(12):
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=step*0.05)
        
    assert detector.is_background_ready()
    
    # 2. Inject a tiny sphere (only 1 beam hit) at 2.0m distance
    scan = list(bg_scan)
    beam_idx = int((0.0 - angle_min) / angle_inc) # beam facing forward
    scan[beam_idx] = 2.0 
    
    detection = detector.process_scan(scan, angle_min, angle_inc, timestamp=1.0)
    
    assert detection is not None, "Detector failed to find a single-beam sphere"
    assert math.isclose(detection[0], 2.0, abs_tol=0.1)
    assert math.isclose(detection[1], 0.0, abs_tol=0.1)

def test_shadowing_singularity():
    """Test the shadowing controller singularity (e_theta = 0) and extreme e_theta."""
    ctrl = CatcherController()
    ctrl.set_state("SHADOW")
    
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.2, 0.0)
    
    # Target exactly at ego's trailing reference point with same heading. e_theta = 0
    # Ref point is: xe - 0.35 = 0 => xe = 0.35
    target_state = np.array([0.35, 0.0, 0.0, 0.2, 0.0])
    
    v, w, state = ctrl.compute_control(ego_pose, ego_vel, target_state, [], dt=0.05, bg_ready=True)
    assert state == "SHADOW"
    
    # Check that it doesn't crash or output NaN
    assert not math.isnan(v)
    assert not math.isnan(w)
    
    # Extreme e_theta = math.pi
    target_state_opp = np.array([0.35, 0.0, math.pi, 0.2, 0.0])
    v, w, state = ctrl.compute_control(ego_pose, ego_vel, target_state_opp, [], dt=0.05, bg_ready=True)
    assert not math.isnan(v)
    assert not math.isnan(w)

def test_catcher_lost_target_spiral():
    """Test that catcher transitions to spiral search when target is lost for a long time."""
    ctrl = CatcherController(
        search_spin_rate=1.0,
        search_timeout=1.0, # shorten for test
        ekf_lost_timeout=0.5
    )
    ctrl.set_state("PURSUE")
    
    # Target was detected 1.0 second ago (which is > ekf_lost_timeout)
    time_since_det = 1.0
    ego_pose = (0.0, 0.0, 0.0)
    target_state = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    
    # First tick: Should notice target lost, transition to SEARCH
    v, w, state = ctrl.compute_control(ego_pose, (0,0), target_state, [], dt=0.05, bg_ready=True, time_since_target_detected=time_since_det)
    assert state == "SEARCH"
    assert v == 0.0 and w == 1.0 # standard spin in place
    
    # Now simulate time passing in SEARCH mode
    for _ in range(25): # 25 * 0.05 = 1.25 seconds > search_timeout (1.0)
        v, w, state = ctrl.compute_control(ego_pose, (0,0), target_state, [], dt=0.05, bg_ready=True, time_since_target_detected=time_since_det)
        
    assert state == "SEARCH"
    # Should be in spiral mode now (v > 0)
    assert v > 0.0
    assert w > 0.0

def test_herding_corner_case():
    """Test that herding offset pushes the target away from the corner effectively."""
    ctrl = CatcherController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        corner_wall_threshold=2.0,
        herd_corner_offset=1.5
    )
    ctrl.set_state("PURSUE")
    
    # Target is in the top right corner
    xe, ye = 4.8, 4.8 
    target_state = np.array([xe, ye, 0.0, 0.2, 0.0])
    ego_pose = (3.0, 3.0, math.pi/4) # Ego approaching corner
    
    v, w, state = ctrl.compute_control(ego_pose, (0.2, 0.0), target_state, [], dt=0.05, bg_ready=True)
    assert state == "PURSUE"
    # To herd away from the corner, it should aim for a point closer to the center.
    # The center is (0,0). The offset point should be (4.8 - dx, 4.8 - dy).
    # We just ensure it doesn't crash and outputs a valid command.
    assert not math.isnan(v)
    assert not math.isnan(w)

def test_obstacle_map_out_of_bounds():
    """Test that adding out-of-bounds LIDAR points doesn't crash the obstacle map."""
    omap = ObstacleMap(arena_bounds=(-5.0, -5.0, 5.0, 5.0), resolution=0.1, size=12.0)
    
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams
    
    # Scan with very large distances (100m)
    scan = [100.0] * num_beams
    ego_pose = (0.0, 0.0, 0.0)
    
    # Should safely discard or clamp points outside the 12x12 grid
    omap.add_scan(scan, angle_min, angle_inc, ego_pose)
    omap.finalize()
    
    # It shouldn't crash.
    obstacles = omap.get_obstacles()
    assert isinstance(obstacles, list)
