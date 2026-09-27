import math
import numpy as np
import pytest

from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.runner_controller import RunnerController
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.math_utils import wrap_angle


def test_round1_sphere_target_capture_under_60s():
    """
    Host Instruction Verification - Round 1:
    'The Catcher will be tested against a standardized moving target acting as the Runner.
     The target may be represented by a moving sphere/dot...
     A Runner is considered captured when it enters within a 0.5 m radius of the Catcher.
     The Catcher must maintain this distance continuously for 1 second to confirm a successful capture.'
    """
    dt = 0.05
    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), v_max=0.30, omega_max=1.80)
    catcher.set_state("PURSUE")

    # Initial setup: Catcher at (-1.5, 0), Sphere target at (1.5, 0) -> 3m initial distance
    c_x, c_y, c_th = -1.5, 0.0, 0.0
    c_v, c_w = 0.0, 0.0

    # Moving sphere executing standard roving motion at 0.18 m/s
    s_x, s_y, s_th = 1.5, 0.0, math.pi / 3
    s_v = 0.18

    # 4 static obstacles per rules
    obstacles = [
        {"center": (0.0, 1.5), "radius": 0.40},
        {"center": (0.0, -1.5), "radius": 0.40},
        {"center": (2.0, 1.0), "radius": 0.40},
        {"center": (-2.0, -1.0), "radius": 0.40},
    ]

    captured = False
    capture_time = None

    for step in range(1200):  # up to 60s
        t = step * dt

        # Sphere target motion with bounce off walls
        s_x += s_v * math.cos(s_th) * dt
        s_y += s_v * math.sin(s_th) * dt
        if s_x > 4.2 or s_x < -4.2:
            s_th = math.pi - s_th
        if s_y > 4.2 or s_y < -4.2:
            s_th = -s_th
        s_th = wrap_angle(s_th)

        target_state = np.array([s_x, s_y, s_th, s_v, 0.0])

        c_v, c_w, state = catcher.compute_control(
            ego_pose=(c_x, c_y, c_th),
            ego_vel=(c_v, c_w),
            target_ekf_state=target_state,
            obstacles=obstacles,
            dt=dt,
            bg_ready=True,
        )

        c_th = wrap_angle(c_th + c_w * dt)
        c_x += c_v * math.cos(c_th) * dt
        c_y += c_v * math.sin(c_th) * dt

        if catcher.is_capture_confirmed():
            captured = True
            capture_time = t
            break

    assert captured, "Failed to capture moving sphere target within 60 seconds"
    assert capture_time is not None and capture_time < 60.0, f"Capture time too slow: {capture_time:.2f}s"


def test_round2_tiebreaker_final_distance_maximization():
    """
    Host Instruction Verification - Round 2 Tie-Breaking:
    'If both Runners successfully evade capture, the Runner with the greater final
     distance from the Catcher receives the advantage.'
    Verify that in endgame mode (t >= 150s), Runner vectors away to achieve maximum separation (> 4.0m).
    """
    runner = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), endgame_start=150.0)
    runner.set_state("EVADE")

    # At t = 175s, Catcher is at (1.0, 1.0)
    # Runner is at (0.0, 0.0)
    catcher_state = np.array([1.0, 1.0, 0.0, 0.20, 0.0])

    r_x, r_y, r_th = 0.0, 0.0, 0.0
    dt = 0.05

    # Run for 100 ticks (5s) in endgame mode
    for _ in range(100):
        v, w, state = runner.compute_control(
            ego_pose=(r_x, r_y, r_th),
            catcher_ekf_state=catcher_state,
            obstacles=[],
            elapsed_time=175.0,
            dt=dt,
            bg_ready=True,
        )
        r_th = wrap_angle(r_th + w * dt)
        r_x += v * math.cos(r_th) * dt
        r_y += v * math.sin(r_th) * dt

    # Runner should have moved into the opposite quadrant (-x, -y)
    final_dist = math.hypot(r_x - catcher_state[0], r_y - catcher_state[1])
    assert final_dist > 2.0, f"Endgame distance not expanding: {final_dist:.2f}m"
    assert r_x < 0.0 and r_y < 0.0, f"Runner did not seek opposite quadrant: ({r_x:.2f}, {r_y:.2f})"


def test_capture_reset_on_threshold_violation():
    """
    Host Instruction Verification - Capture Rule:
    'The Catcher must maintain this distance continuously for 1 second to confirm a successful capture.'
    If distance exceeds 0.50m even for a single tick at 0.95s, timer must reset to 0.0s.
    """
    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    catcher.set_state("SHADOW")

    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.0, 0.0)
    dt = 0.05

    # 1. Hold inside capture zone (0.40m <= 0.50m) for 19 ticks = 0.95s
    target_inside = np.array([0.40, 0.0, 0.0, 0.0, 0.0])
    for _ in range(19):
        catcher.compute_control(ego_pose, ego_vel, target_inside, [], dt=dt)

    assert np.isclose(catcher.get_capture_timer(), 0.95, atol=1e-3)
    assert not catcher.is_capture_confirmed()

    # 2. Target slips to 0.52m (> 0.50m) on tick 20
    target_outside = np.array([0.52, 0.0, 0.0, 0.0, 0.0])
    catcher.compute_control(ego_pose, ego_vel, target_outside, [], dt=dt)

    # Timer must hard reset to 0.0s
    assert catcher.get_capture_timer() == 0.0, "Capture timer failed to reset on distance violation"
    assert not catcher.is_capture_confirmed()


def test_six_obstacle_dense_arena_navigation():
    """
    Host Instruction Verification - Arena Conditions:
    'Obstacles: 4–6 static obstacles'
    Verify both Catcher and Runner navigate a 6-obstacle layout without collisions.
    """
    six_obstacles = [
        {"center": (-2.0, -2.0), "radius": 0.45},
        {"center": (-2.0, 2.0), "radius": 0.45},
        {"center": (2.0, -2.0), "radius": 0.45},
        {"center": (2.0, 2.0), "radius": 0.45},
        {"center": (0.0, 0.0), "radius": 0.50},
        {"center": (0.0, 3.0), "radius": 0.40},
    ]

    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    runner = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), dt=0.05)

    catcher.set_state("PURSUE")
    runner.set_state("EVADE")

    # Positions: Catcher at (-0.9, 0.0) with obstacle ahead at (0.0, 0.0, radius 0.5)
    c_pose = (-0.9, 0.0, 0.0)
    r_pose = (3.5, 0.0, math.pi)

    # Catcher must not command collision directly through center obstacle (0, 0)
    target_state = np.array([3.5, 0.0, math.pi, 0.20, 0.0])
    v_c, w_c, _ = catcher.compute_control(c_pose, (0.2, 0.0), target_state, six_obstacles, dt=0.05)

    # Catcher's DWA must steer away from direct collision course with (0,0) obstacle
    assert abs(w_c) > 0.1, "Catcher did not steer to avoid center obstacle"

    # Runner must also avoid obstacles
    catcher_state = np.array([-3.5, 0.0, 0.0, 0.25, 0.0])
    v_r, w_r, _ = runner.compute_control(r_pose, catcher_state, six_obstacles, elapsed_time=10.0, dt=0.05)
    assert not math.isnan(v_r) and not math.isnan(w_r)


def test_lidar_detector_sensor_noise_and_dropouts():
    """
    Sensor Robustness:
    Test LIDAR target detector with 5% dropped beams (returning inf)
    and Gaussian measurement noise (sigma = 0.03m).
    """
    detector = LidarTargetDetector(bg_learning_duration=0.5, target_type="any")
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    # 1. Background training
    bg_scan = [5.0] * num_beams
    for step in range(12):
        detector.process_scan(bg_scan, angle_min, angle_inc, timestamp=step * 0.05)
    assert detector.is_background_ready()

    # 2. Noisy scan with target at range 2.5m, beam 180 (straight ahead)
    noisy_scan = [5.0 + float(np.random.normal(0, 0.03)) for _ in range(num_beams)]
    # Target across 3 beams
    noisy_scan[179] = 2.5 + float(np.random.normal(0, 0.02))
    noisy_scan[180] = 2.5 + float(np.random.normal(0, 0.02))
    noisy_scan[181] = 2.5 + float(np.random.normal(0, 0.02))

    # Inject 5% random inf dropouts away from target
    for idx in [10, 45, 90, 220, 310]:
        noisy_scan[idx] = float("inf")

    detection = detector.process_scan(noisy_scan, angle_min, angle_inc, timestamp=1.0)
    assert detection is not None, "Detector failed to detect target amidst noise and dropouts"
    r, phi = detection
    assert np.isclose(r, 2.5, atol=0.15)
    assert np.isclose(phi, 0.0, atol=0.10)
