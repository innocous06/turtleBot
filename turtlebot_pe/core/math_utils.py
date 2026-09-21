import math
import numpy as np


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def euclidean_distance(p1, p2) -> float:
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return float(math.sqrt(dx * dx + dy * dy))


def unit_vector(vec, eps: float = 1e-6) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def clamp(val: float, min_val: float, max_val: float) -> float:
    return float(max(min_val, min(max_val, val)))


def rotate_2d(vec, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]]) @ np.array(vec, dtype=float)
