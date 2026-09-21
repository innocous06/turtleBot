"""
Unit tests for CatcherController.
"""

import math
import numpy as np
import pytest
from turtlebot_pe.core.catcher_controller import CatcherController


def test_catcher_init_to_search():
    ctrl = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    assert ctrl.get_state() == "INIT"

    # With bg_ready=False, remains INIT
    v, w, state = ctrl.compute_control((0, 0, 0), (0, 0), None, [], dt=0.05, bg_ready=False)
    assert state == "INIT"
    assert v == 0.0 and w == 0.0

    # With bg_ready=True, transitions to SEARCH
    v, w, state = ctrl.compute_control((0, 0, 0), (0, 0), None, [], dt=0.05, bg_ready=True)
    assert state == "SEARCH"
    assert w > 0.0  # Spinning search


def test_catcher_pursue_proportional_navigation():
    ctrl = CatcherController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    ctrl.set_state("PURSUE")

    # Ego at origin facing +x (0 deg). Target at (3.0, 1.0) moving upwards (+y)
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.2, 0.0)
    target_state = np.array([3.0, 1.0, math.pi / 2, 0.20, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        ego_vel=ego_vel,
        target_ekf_state=target_state,
        obstacles=[],
        dt=0.05,
        bg_ready=True,
    )

    assert state == "PURSUE"
    # Target is to the left (+y), so steering command w_cmd must be positive (turn left)
    assert w_cmd > 0.0
    assert 0.0 < v_cmd <= ctrl.v_max


def test_catcher_shadow_transition_and_capture_timer():
    ctrl = CatcherController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        shadow_enter_distance=0.70,
        capture_radius=0.50,
        capture_hold_target=1.0,
    )
    ctrl.set_state("PURSUE")

    # Target is 0.40m directly in front of ego -> within capture radius
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.2, 0.0)
    target_state = np.array([0.40, 0.0, 0.0, 0.20, 0.0])

    v, w, state = ctrl.compute_control(ego_pose, ego_vel, target_state, [], dt=0.05)
    assert state == "SHADOW"
    assert ctrl.get_capture_timer() > 0.0

    # Hold for 1 second (20 steps of dt=0.05)
    for _ in range(21):
        v, w, state = ctrl.compute_control(ego_pose, ego_vel, target_state, [], dt=0.05)

    assert ctrl.is_capture_confirmed()
    assert state == "CAPTURED"
    assert v == 0.0 and w == 0.0


def test_catcher_shadow_reset_on_escape():
    ctrl = CatcherController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        shadow_enter_distance=0.70,
        shadow_exit_distance=0.80,
        capture_radius=0.50,
    )
    ctrl.set_state("SHADOW")
    ego_pose = (0.0, 0.0, 0.0)
    ego_vel = (0.2, 0.0)

    # Within capture radius
    target_close = np.array([0.40, 0.0, 0.0, 0.2, 0.0])
    ctrl.compute_control(ego_pose, ego_vel, target_close, [], dt=0.05)
    assert ctrl.get_capture_timer() > 0.0

    # Target pulls away beyond capture radius (0.60m)
    target_mid = np.array([0.60, 0.0, 0.0, 0.2, 0.0])
    ctrl.compute_control(ego_pose, ego_vel, target_mid, [], dt=0.05)
    assert ctrl.get_capture_timer() == 0.0  # Reset
    assert ctrl.get_state() == "SHADOW"

    # Target escapes past exit distance (0.90m)
    target_far = np.array([0.90, 0.0, 0.0, 0.2, 0.0])
    ctrl.compute_control(ego_pose, ego_vel, target_far, [], dt=0.05)
    assert ctrl.get_state() == "PURSUE"
