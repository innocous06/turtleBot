import math
import numpy as np
import pytest

from turtlebot_pe.core.catcher_controller import CatcherController
from turtlebot_pe.core.runner_controller import RunnerController
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.math_utils import wrap_angle


def test_runner_corner_escape_not_deadlocked():
    """Verify Runner does not get deadlocked when trapped in a corner by pursuer."""
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Runner trapped at (4.7, 4.7) in corner.
    # Catcher approaching from (3.8, 3.8) directly along the diagonal.
    ego_pose = (4.7, 4.7, -math.pi / 2)  # Facing south
    catcher_state = np.array([3.8, 3.8, math.pi / 4, 0.30, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=30.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "EVADE"
    # Velocity must not freeze to 0; runner must actively steer away
    assert abs(w_cmd) > 0.0
    assert not math.isnan(v_cmd)
    assert not math.isnan(w_cmd)


def test_runner_collinear_obstacle_vortex():
    """Verify vortex field breaks collinear deadlock when obstacle is exactly between robots."""
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Perfectly collinear along y=0: Catcher at (0, 0), Obstacle at (2, 0), Runner at (3, 0)
    obstacles = [{"center": (2.0, 0.0), "radius": 0.4}]
    ego_pose = (3.0, 0.0, 0.0)
    catcher_state = np.array([0.0, 0.0, 0.0, 0.30, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=obstacles,
        elapsed_time=20.0,
        dt=0.05,
        bg_ready=True,
    )

    # Must produce lateral motion (angular command != 0) to slip past obstacle
    assert abs(w_cmd) > 0.05, "Collinear obstacle created deadlock without turning"


def test_catcher_shadow_target_sudden_stop():
    """Verify Catcher stabilizes behind a target that abruptly stops from full speed."""
    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    catcher.set_state("SHADOW")

    # Ego initially trailing target at 0.45m
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.28, 0.0)
    
    # Target suddenly at v=0, located at (0.45, 0)
    target_state = np.array([0.45, 0.0, 0.0, 0.0, 0.0])

    # Run 10 control ticks (0.5s)
    curr_x = ego_pose[0]
    for _ in range(10):
        v, w, state = catcher.compute_control(
            ego_pose=(curr_x, 0.0, 0.0),
            ego_vel=ego_vel,
            target_ekf_state=target_state,
            obstacles=[],
            dt=0.05,
        )
        curr_x += v * 0.05
        # Must maintain safety margin: cannot ram target (collision distance = 0.34m)
        dist_to_target = 0.45 - curr_x
        assert dist_to_target > 0.15, f"Catcher rammed into stopped target! dist={dist_to_target}"


def test_catcher_shadow_target_sharp_turn():
    """Verify Catcher tracks a target taking a 90-degree turn without losing shadow."""
    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    catcher.set_state("SHADOW")

    # Target turns 90 degrees (+y)
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.25, 0.0)
    target_state = np.array([0.40, 0.0, math.pi / 2, 0.25, 0.5])

    v, w, state = catcher.compute_control(ego_pose, ego_vel, target_state, [], dt=0.05)

    assert state == "SHADOW"
    # Catcher must turn left (+w) to follow the target's heading
    assert w > 0.5, "Catcher failed to steer into target's sharp turn"


def test_full_match_simulation_arena():
    """
    Simulate a closed-loop 1v1 match between CatcherController and RunnerController
    in the 10x10 arena with 5 static obstacles for 60 seconds (1200 ticks).
    """
    dt = 0.05
    max_steps = 1200  # 60s simulation

    obstacles = [
        {"center": (2.0, 2.0), "radius": 0.45},
        {"center": (-2.0, 1.5), "radius": 0.45},
        {"center": (1.0, -2.0), "radius": 0.45},
        {"center": (-1.5, -1.0), "radius": 0.45},
        {"center": (0.0, 2.5), "radius": 0.45},
    ]

    catcher = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    runner = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), dt=dt)

    catcher.set_state("PURSUE")
    runner.set_state("EVADE")

    # Initial spawn positions 3m apart: catcher at (-1.5, 0), runner at (1.5, 0)
    c_x, c_y, c_th = -1.5, 0.0, 0.0
    c_v, c_w = 0.0, 0.0

    r_x, r_y, r_th = 1.5, 0.0, 0.0
    r_v, r_w = 0.0, 0.0

    captured = False
    capture_time = None

    for step in range(max_steps):
        t = step * dt

        # Target states
        target_for_catcher = np.array([r_x, r_y, r_th, r_v, r_w])
        target_for_runner = np.array([c_x, c_y, c_th, c_v, c_w])

        # Compute commands
        c_v, c_w, c_state = catcher.compute_control(
            ego_pose=(c_x, c_y, c_th),
            ego_vel=(c_v, c_w),
            target_ekf_state=target_for_catcher,
            obstacles=obstacles,
            dt=dt,
            bg_ready=True,
        )

        r_v, r_w, r_state = runner.compute_control(
            ego_pose=(r_x, r_y, r_th),
            catcher_ekf_state=target_for_runner,
            obstacles=obstacles,
            elapsed_time=t,
            dt=dt,
            bg_ready=True,
        )

        # Kinematic update for Catcher
        if abs(c_w) >= 1e-4:
            c_th = wrap_angle(c_th + c_w * dt)
            c_x += (c_v / c_w) * (math.sin(c_th) - math.sin(c_th - c_w * dt))
            c_y += (c_v / c_w) * (-math.cos(c_th) + math.cos(c_th - c_w * dt))
        else:
            c_x += c_v * math.cos(c_th) * dt
            c_y += c_v * math.sin(c_th) * dt

        # Kinematic update for Runner
        if abs(r_w) >= 1e-4:
            r_th = wrap_angle(r_th + r_w * dt)
            r_x += (r_v / r_w) * (math.sin(r_th) - math.sin(r_th - r_w * dt))
            r_y += (r_v / r_w) * (-math.cos(r_th) + math.cos(r_th - r_w * dt))
        else:
            r_x += r_v * math.cos(r_th) * dt
            r_y += r_v * math.sin(r_th) * dt

        # Check bounds: neither robot should leave the 10x10 arena [-5, 5]
        assert -5.1 <= c_x <= 5.1 and -5.1 <= c_y <= 5.1, f"Catcher escaped arena at ({c_x}, {c_y})"
        assert -5.1 <= r_x <= 5.1 and -5.1 <= r_y <= 5.1, f"Runner escaped arena at ({r_x}, {r_y})"

        dist = math.hypot(c_x - r_x, c_y - r_y)
        if catcher.is_capture_confirmed():
            captured = True
            capture_time = t
            break

    # The simulation should execute cleanly without exceptions
    assert captured or t >= 59.0
