import math
import numpy as np
import pytest
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.runner_controller import RunnerController
from turtlebot_pe.core.obstacle_map import ObstacleMap


def test_catcher_pipeline_end_to_end():
    dt = 0.05
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    detector = LidarTargetDetector(bg_learning_duration=0.5, target_type="any")
    ekf = TargetEKF(dt=dt)
    controller = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    obstacle_map = ObstacleMap(arena_bounds=(-5.0, -5.0, 5.0, 5.0))

    ego_pose = (0.0, 0.0, 0.0)
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

    # 2. Simulate target moving towards catcher
    simulated_target_x = 3.0
    for step in range(30):
        t = 1.0 + step * dt
        simulated_target_x -= 0.02

        scan = list(bg_scan)
        beam_idx = int((0.0 - angle_min) / angle_inc)
        scan[beam_idx - 1] = simulated_target_x
        scan[beam_idx] = simulated_target_x
        scan[beam_idx + 1] = simulated_target_x

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

        assert 0.0 <= v_cmd <= controller.v_max
        assert -controller.omega_max <= w_cmd <= controller.omega_max
        assert state in ["SEARCH", "PURSUE", "SHADOW", "CAPTURED"]


def test_runner_pipeline_end_to_end():
    dt = 0.05
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    detector = LidarTargetDetector(bg_learning_duration=0.5, target_type="robot")
    ekf = TargetEKF(dt=dt)
    controller = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), dt=dt)
    obstacle_map = ObstacleMap(arena_bounds=(-5.0, -5.0, 5.0, 5.0))

    runner_pose = (1.5, 0.0, 0.0)

    # 1. Train background
    wall_dist = 5.0
    bg_scan = [wall_dist] * num_beams
    for step in range(12):
        t = step * dt
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=t)
        obstacle_map.add_scan(bg_scan, angle_min, angle_inc, runner_pose)

    assert detector.is_background_ready()
    obstacle_map.finalize()

    # 2. Simulate catcher robot (width ~0.35m) pursuing from behind
    catcher_x = -1.0
    for step in range(30):
        t = 1.0 + step * dt
        catcher_x += 0.015
        dist = 1.5 - catcher_x

        scan = list(bg_scan)
        # Behind runner is angle pi radians; at dist=2.5m, 0.35m width is ~8 beams
        beam_idx = int((math.pi - angle_min) / angle_inc) % num_beams
        for offset in range(-4, 5):
            scan[(beam_idx + offset) % num_beams] = dist

        ekf.predict()
        detection = detector.process_scan(scan, angle_min, angle_inc, timestamp=t)
        assert detection is not None

        accepted, _ = ekf.update(detection, runner_pose)
        assert accepted

        catcher_state = ekf.get_state()
        v_cmd, w_cmd, state = controller.compute_control(
            ego_pose=runner_pose,
            catcher_ekf_state=catcher_state,
            obstacles=obstacle_map.get_obstacles(),
            elapsed_time=t,
            dt=dt,
            bg_ready=True,
        )

        assert 0.0 <= v_cmd <= controller.v_max
        assert -controller.omega_max <= w_cmd <= controller.omega_max
        assert state in ["EVADE", "SHIELD", "SURVIVED"]
