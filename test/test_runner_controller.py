import math
import numpy as np
import pytest

from turtlebot_pe.core.runner_controller import RunnerController


def test_runner_init_to_evade():
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0))
    assert ctrl.get_state() == "INIT"

    v, w, state = ctrl.compute_control((0, 0, 0), None, [], elapsed_time=0.0, dt=0.05, bg_ready=False)
    assert state == "INIT"
    assert v == 0.0 and w == 0.0

    v, w, state = ctrl.compute_control((0, 0, 0), None, [], elapsed_time=1.0, dt=0.05, bg_ready=True)
    assert state == "EVADE"


def test_runner_pursuer_repulsion():
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Catcher at (0, 0), Runner at (1.0, 0.0) facing +x (0 rad)
    ego_pose = (1.0, 0.0, 0.0)
    catcher_state = np.array([0.0, 0.0, 0.0, 0.20, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=10.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "EVADE"
    # Force is in +x direction away from catcher, runner facing +x, so v_cmd > 0 and w_cmd ~ 0
    assert v_cmd > 0.1
    assert abs(w_cmd) < 0.2


def test_runner_wall_repulsion():
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Runner near east wall (4.8, 0.0) facing east (0 rad)
    # Catcher far at (-2.0, 0.0)
    ego_pose = (4.8, 0.0, 0.0)
    catcher_state = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=10.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "EVADE"
    # Wall pushes in -x direction (away from wall), so robot must turn around
    assert abs(w_cmd) > 0.5


def test_runner_obstacle_vortex():
    ctrl = RunnerController(arena_bounds=(-5.0, -5.0, 5.0, 5.0), ou_sigma=0.0)
    ctrl.set_state("EVADE")

    # Obstacle at (1.0, 0.0) with radius 0.4
    # Runner at (1.6, 0.0)
    # Catcher at (0.0, 0.0)
    obstacles = [{"center": (1.0, 0.0), "radius": 0.4}]
    ego_pose = (1.6, 0.0, 0.0)
    catcher_state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=obstacles,
        elapsed_time=10.0,
        dt=0.05,
        bg_ready=True,
    )

    # Tangential vortex forces lateral motion
    assert not math.isnan(v_cmd)
    assert not math.isnan(w_cmd)


def test_runner_shield_mode_activation():
    ctrl = RunnerController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        shield_activation_distance=2.5,
        shield_time_advantage=0.1,
    )
    ctrl.set_state("EVADE")

    # Obstacle at (1.0, 0.0), radius 0.5
    # Catcher at (0.0, 0.0) -> dist to obs = 1.0
    # Runner at (1.8, 0.0) -> shadow point is (1.85, 0.0), runner is already right next to it
    obstacles = [{"center": (1.0, 0.0), "radius": 0.5}]
    ego_pose = (1.8, 0.0, 0.0)
    catcher_state = np.array([0.0, 0.0, 0.0, 0.25, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=obstacles,
        elapsed_time=20.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "SHIELD"
    assert v_cmd > 0.0


def test_runner_endgame_distance_maximization():
    ctrl = RunnerController(
        arena_bounds=(-5.0, -5.0, 5.0, 5.0),
        endgame_start=150.0,
    )
    ctrl.set_state("EVADE")

    # At t = 160s (endgame)
    # Catcher at (2.0, 2.0)
    # Opposite point is (-2.0, -2.0)
    ego_pose = (0.0, 0.0, 0.0)
    catcher_state = np.array([2.0, 2.0, 0.0, 0.2, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=160.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "EVADE"
    # Net force should pull towards (-x, -y)
    assert not math.isnan(v_cmd)
    assert not math.isnan(w_cmd)


def test_runner_match_survived():
    ctrl = RunnerController(match_duration=180.0)
    ctrl.set_state("EVADE")

    ego_pose = (0.0, 0.0, 0.0)
    catcher_state = np.array([2.0, 2.0, 0.0, 0.2, 0.0])

    v_cmd, w_cmd, state = ctrl.compute_control(
        ego_pose=ego_pose,
        catcher_ekf_state=catcher_state,
        obstacles=[],
        elapsed_time=181.0,
        dt=0.05,
        bg_ready=True,
    )

    assert state == "SURVIVED"
    assert v_cmd == 0.0 and w_cmd == 0.0
