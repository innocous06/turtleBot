"""
Unit tests for LidarTargetDetector.
"""

import math
import numpy as np
import pytest
from turtlebot_pe.core.lidar_detector import LidarTargetDetector


def test_background_learning():
    detector = LidarTargetDetector(bg_learning_duration=1.0)
    assert not detector.is_background_ready()

    # Static circular room of 5.0m
    num_beams = 360
    static_scan = [5.0] * num_beams

    # Feed scans at t = 0.0, 0.5, 1.1s
    detector.process_scan(static_scan, -math.pi, 2 * math.pi / num_beams, timestamp=0.0)
    assert not detector.is_background_ready()

    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        detector.process_scan(static_scan, -math.pi, 2 * math.pi / num_beams, timestamp=t)

    assert detector.is_background_ready()


def test_foreground_target_detection():
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    detector = LidarTargetDetector(
        bg_learning_duration=1.0,
        foreground_threshold=0.15,
        min_cluster_size=1,
        min_target_width=0.04,
        target_type="any",
    )

    # Train background at 5.0m
    static_scan = [5.0] * num_beams
    for t in np.linspace(0.0, 1.2, 10):
        detector.process_scan(static_scan, angle_min, angle_inc, timestamp=float(t))
    assert detector.is_background_ready()

    # Place an intruder at beam 180 (angle = 0 radians, straight ahead) at range 2.0m (4 beams wide)
    intruder_scan = list(static_scan)
    intruder_scan[179] = 2.0
    intruder_scan[180] = 2.0
    intruder_scan[181] = 2.0

    detection = detector.process_scan(intruder_scan, angle_min, angle_inc, timestamp=2.0)
    assert detection is not None
    r, phi = detection
    assert np.isclose(r, 2.0, atol=0.1)
    assert np.isclose(phi, 0.0, atol=0.05)


def test_cluster_size_rejection():
    num_beams = 360
    angle_min = -math.pi
    angle_inc = 2 * math.pi / num_beams

    detector = LidarTargetDetector(
        bg_learning_duration=1.0,
        min_cluster_size=3,
        max_cluster_size=20,
        target_type="robot",
    )

    static_scan = [5.0] * num_beams
    for t in np.linspace(0.0, 1.2, 10):
        detector.process_scan(static_scan, angle_min, angle_inc, timestamp=float(t))

    # Single point noise should be rejected when min_cluster_size=3
    noise_scan = list(static_scan)
    noise_scan[90] = 1.5
    detection = detector.process_scan(noise_scan, angle_min, angle_inc, timestamp=2.0)
    assert detection is None
