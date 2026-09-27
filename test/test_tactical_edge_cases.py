import math
import numpy as np
import pytest

from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.runner_controller import RunnerController
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.math_utils import wrap_angle


def test_catcher_intercepts_sinusoidal_target():
    """Verify Catcher closes distance against an evader performing continuous S-curves."""
    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    catcher.set_state("PURSUE")

    # Initial states
    c_x, c_y, c_th = 0.0, 0.0, 0.0
    t_x, t_y = 3.0, 0.0

    dt = 0.05
    initial_dist = math.hypot(t_x - c_x, t_y - c_y)

    for step in range(80):  # 4 seconds
        t = step * dt
        # Target executes sinusoidal trajectory
        t_x += 0.005
        t_y = 0.8 * math.sin(2.0 * t)
        t_th = math.atan2(0.8 * 2.0 * math.cos(2.0 * t), 0.1)
        t_vel = 0.20

        target_state = np.array([t_x, t_y, t_th, t_vel, 0.0])

        v, w, state = catcher.compute_control(
            ego_pose=(c_x, c_y, c_th),
            ego_vel=(0.28, 0.0),
            target_ekf_state=target_state,
            obstacles=[],
            dt=dt,
        )

        c_th = wrap_angle(c_th + w * dt)
        c_x += v * math.cos(c_th) * dt
        c_y += v * math.sin(c_th) * dt

    final_dist = math.hypot(t_x - c_x, t_y - c_y)
    assert final_dist < initial_dist, f"Catcher failed to close distance: {initial_dist:.2f} -> {final_dist:.2f}"


def test_lidar_wrap_around_clustering():
    """Verify clusters spanning across -pi / +pi boundary are merged correctly."""
    detector = LidarTargetDetector(
        bg_learning_duration=0.5,
        target_type="any",
        min_cluster_size=2,
        min_target_width=0.04,
    )

    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    # 1. Train background
    bg_scan = [5.0] * num_beams
    for step in range(12):
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=step * 0.05)
    assert detector.is_background_ready()

    # 2. Target sits directly behind (-pi / +pi boundary)
    # Beams 0 (at -pi) and beam 359 (at +pi - delta)
    scan = list(bg_scan)
    scan[0] = 2.0
    scan[1] = 2.0
    scan[358] = 2.0
    scan[359] = 2.0

    detection = detector.process_scan(scan, angle_min, angle_inc, timestamp=1.0)
    assert detection is not None, "Detector failed to detect wrap-around cluster"
    r, phi = detection
    assert np.isclose(r, 2.0, atol=0.1)
    # Bearing should be near -pi or +pi
    assert abs(abs(phi) - math.pi) < 0.15


def test_target_loss_and_reacquisition():
    """Verify EKF inflates covariance during 2s blind period and updates immediately upon recovery."""
    ekf = TargetEKF(dt=0.05)
    ekf.init_state(x=2.0, y=0.0, theta=0.0, v=0.20, omega=0.0)

    initial_trace = np.trace(ekf.get_covariance())

    # Blind period: 40 ticks = 2.0s without detection
    for _ in range(40):
        ekf.predict()
        ekf.handle_occlusion()

    blind_trace = np.trace(ekf.get_covariance())
    assert blind_trace > initial_trace * 2.0, "Covariance did not inflate sufficiently during occlusion"

    # Re-acquisition: new measurement arrives
    ego_pose = (0.0, 0.0, 0.0)
    # Predicted position moved forward: 2.0 + 0.2 * 2.0 = 2.4m
    meas = [2.42, 0.0]
    accepted, d_m2 = ekf.update(meas, ego_pose)

    assert accepted, f"EKF failed to re-acquire target after occlusion (d_m2={d_m2})"
    recovered_trace = np.trace(ekf.get_covariance())
    assert recovered_trace < blind_trace, "Covariance did not reduce upon re-acquisition"


def test_runner_wall_hugging_evasion():
    """Verify Runner being chased parallel to wall maintains high forward velocity away from pursuer."""
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Runner at (0.0, 4.3) moving east (+x), 0.7m from north wall
    # Catcher trailing at (-1.5, 4.3)
    ego_pose = (0.0, 4.3, 0.0)
    catcher_state = np.array([-1.5, 4.3, 0.0, 0.30, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=20.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "EVADE"
    # Should sprint forward (+x) with max or near-max speed
    assert v_cmd > 0.20, f"Runner slowed down excessively near wall: v={v_cmd}"
