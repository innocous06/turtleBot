"""
Unit tests for TargetEKF (CTRV model).
Can be run standalone with pytest.
"""

import math
import numpy as np
import pytest
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.math_utils import wrap_angle


def test_ekf_initialization():
    ekf = TargetEKF(dt=0.05)
    assert not ekf.is_initialized()
    ekf.init_state(x=1.0, y=2.0, theta=0.5, v=0.2, omega=0.1)
    assert ekf.is_initialized()
    state = ekf.get_state()
    assert np.isclose(state[0], 1.0)
    assert np.isclose(state[1], 2.0)
    assert np.isclose(state[2], 0.5)
    assert np.isclose(state[3], 0.2)
    assert np.isclose(state[4], 0.1)


def test_ekf_straight_line_predict():
    ekf = TargetEKF(dt=0.10)
    # Target moving along X axis at 0.30 m/s with zero turn rate
    ekf.init_state(x=0.0, y=0.0, theta=0.0, v=0.30, omega=0.0)
    pred_state = ekf.predict()
    assert np.isclose(pred_state[0], 0.03, atol=1e-4)
    assert np.isclose(pred_state[1], 0.0, atol=1e-4)
    assert np.isclose(pred_state[2], 0.0, atol=1e-4)


def test_ekf_turning_predict():
    ekf = TargetEKF(dt=0.10)
    # Target turning at 0.5 rad/s with speed 0.2 m/s
    ekf.init_state(x=0.0, y=0.0, theta=0.0, v=0.20, omega=0.5)
    pred_state = ekf.predict()
    expected_th = 0.05
    assert np.isclose(pred_state[2], expected_th, atol=1e-4)
    assert pred_state[0] > 0.0
    assert pred_state[1] > 0.0


def test_ekf_measurement_update_convergence():
    ekf = TargetEKF(dt=0.05)
    # True target is at (3.0, 0.0) relative to ego at (0.0, 0.0, 0.0)
    ego_pose = (0.0, 0.0, 0.0)
    # First measurement initializes the filter
    accepted, _ = ekf.update([3.0, 0.0], ego_pose)
    assert accepted
    assert np.isclose(ekf.get_position()[0], 3.0, atol=0.05)
    assert np.isclose(ekf.get_position()[1], 0.0, atol=0.05)

    # Set a higher initial uncertainty to verify that updates reduce covariance
    ekf.P = np.eye(5) * 2.0
    init_cov_norm = np.linalg.norm(ekf.get_covariance())
    for _ in range(10):
        ekf.predict()
        accepted, d_m2 = ekf.update([3.0, 0.0], ego_pose)
        assert accepted

    final_cov_norm = np.linalg.norm(ekf.get_covariance())
    assert final_cov_norm < init_cov_norm
    assert np.isclose(ekf.get_position()[0], 3.0, atol=0.05)
    assert np.isclose(ekf.get_position()[1], 0.0, atol=0.05)


def test_ekf_mahalanobis_outlier_rejection():
    ekf = TargetEKF(dt=0.05)
    ego_pose = (0.0, 0.0, 0.0)
    ekf.init_state(x=3.0, y=0.0, theta=0.0, v=0.0, omega=0.0)

    # Wild outlier measurement at r=9.0, bearing=pi/2
    accepted, d_m2 = ekf.update([9.0, math.pi / 2], ego_pose)
    assert not accepted
    assert d_m2 > 5.991


def test_ekf_occlusion_inflation():
    ekf = TargetEKF(dt=0.05)
    ekf.init_state(x=2.0, y=1.0, theta=0.0, v=0.2, omega=0.0)
    P_initial = ekf.get_covariance().copy()

    # Occlusion handling should inflate covariance
    for _ in range(10):
        ekf.predict()
        ekf.handle_occlusion()

    P_occluded = ekf.get_covariance()
    assert np.all(np.diag(P_occluded) > np.diag(P_initial))


def test_angle_wrapping():
    assert np.isclose(wrap_angle(math.pi * 3), math.pi, atol=1e-5)
    assert np.isclose(wrap_angle(-math.pi * 3), -math.pi, atol=1e-5)
    assert -math.pi <= wrap_angle(5.0) <= math.pi
