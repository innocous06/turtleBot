"""
Integration test for full perception and pursuit pipeline:
LidarTargetDetector -> TargetEKF -> CatcherController.
"""

import math
import numpy as np
import pytest
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.obstacle_map import ObstacleMap


def test_full_pipeline_end_to_end():
    dt = 0.05
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    detector = LidarTargetDetector(bg_learning_duration=0.5, target_type="any")
    ekf = TargetEKF(dt=dt)
    controller = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    obstacle_map = ObstacleMap(arena_bounds=(-5.0, -5.0, 5.0, 5.0))

    ego_pose = (0.0, 0.0, 0.0)  # Catcher at (0, 0)
    ego_vel = (0.0, 0.0)

    # 1. Train background
    wall_dist = 5.0
    bg_scan = [wall_dist] * num_beams
    for step in range(12):
        t = step * dt
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=t)
        obstacle_map.add_scan(bg_scan, angle_min, angle_inc, ego_pose)

    assert detector.is_background_ready()
    obstacle_map.finalize()

    # 2. Simulate target moving across the field from (3.0, 0.0) to (2.0, 0.0)
    simulated_target_x = 3.0
    for step in range(30):
        t = 1.0 + step * dt
        simulated_target_x -= 0.02  # moving towards catcher

        # Generate scan with target return at beam 180 (ahead)
        scan = list(bg_scan)
        beam_idx = int((0.0 - angle_min) / angle_inc)
        scan[beam_idx - 1] = simulated_target_x
        scan[beam_idx] = simulated_target_x
        scan[beam_idx + 1] = simulated_target_x

        # Pipeline execution
        ekf.predict()
        detection = detector.process_scan(scan, angle_min, angle_inc, timestamp=t)
        assert detection is not None

        accepted, _ = ekf.update(detection, ego_pose)
        assert accepted

        target_state = ekf.get_state()
        v_cmd, w_cmd, state = controller.compute_control(
            ego_pose=ego_pose,
            ego_vel=ego_vel,
            target_ekf_state=target_state,
            obstacles=obstacle_map.get_obstacles(),
            dt=dt,
            bg_ready=True,
        )

        # Assert valid control ranges
        assert 0.0 <= v_cmd <= controller.v_max
        assert -controller.omega_max <= w_cmd <= controller.omega_max
        assert state in ["SEARCH", "PURSUE", "SHADOW", "CAPTURED"]
