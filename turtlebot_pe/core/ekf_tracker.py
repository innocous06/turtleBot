import math
import numpy as np
from .math_utils import wrap_angle


class TargetEKF:
    def __init__(
        self,
        dt: float = 0.05,
        sigma_a: float = 0.5,
        sigma_alpha: float = 1.2,
        sigma_r: float = 0.03,
        sigma_phi: float = 0.02,
        confidence_pos_std_thresh: float = 1.5,
    ):
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self.sigma_alpha = float(sigma_alpha)
        self.sigma_r = float(sigma_r)
        self.sigma_phi = float(sigma_phi)
        self.confidence_pos_std_thresh = float(confidence_pos_std_thresh)

        # State [x, y, theta, v, omega]
        self.x = np.zeros(5, dtype=float)
        self.P = np.eye(5, dtype=float) * 0.1
        self.R = np.diag([self.sigma_r**2, self.sigma_phi**2])
        self.Q_occ = np.diag([0.02, 0.02, 0.10, 0.05, 0.20])

        self._is_initialized = False
        self.time_since_last_measurement = 0.0

    def init_state(self, x: float, y: float, theta: float = 0.0, v: float = 0.0, omega: float = 0.0) -> None:
        self.x = np.array([x, y, wrap_angle(theta), v, omega], dtype=float)
        self.P = np.diag([0.05, 0.05, 0.20, 0.10, 0.20])
        self._is_initialized = True
        self.time_since_last_measurement = 0.0

    def is_initialized(self) -> bool:
        return self._is_initialized

    def predict(self) -> np.ndarray:
        if not self._is_initialized:
            return self.x.copy()

        dt = self.dt
        x, y, th, v, om = self.x

        if abs(om) >= 1e-4:
            th_new = wrap_angle(th + om * dt)
            x_new = x + (v / om) * (math.sin(th + om * dt) - math.sin(th))
            y_new = y + (v / om) * (-math.cos(th + om * dt) + math.cos(th))
        else:
            th_new = th
            x_new = x + v * math.cos(th) * dt
            y_new = y + v * math.sin(th) * dt

        self.x = np.array([x_new, y_new, th_new, v, om], dtype=float)

        # Jacobian F
        F = np.eye(5, dtype=float)
        if abs(om) >= 1e-4:
            s_th = math.sin(th)
            c_th = math.cos(th)
            s_new = math.sin(th + om * dt)
            c_new = math.cos(th + om * dt)
            F[0, 2] = (v / om) * (c_new - c_th)
            F[0, 3] = (1.0 / om) * (s_new - s_th)
            F[0, 4] = -(v / (om**2)) * (s_new - s_th) + (v * dt / om) * c_new
            F[1, 2] = (v / om) * (s_new - s_th)
            F[1, 3] = (1.0 / om) * (-c_new + c_th)
            F[1, 4] = -(v / (om**2)) * (-c_new + c_th) + (v * dt / om) * s_new
            F[2, 4] = dt
        else:
            F[0, 2] = -v * math.sin(th) * dt
            F[0, 3] = math.cos(th) * dt
            F[0, 4] = -0.5 * v * (dt**2) * math.sin(th)
            F[1, 2] = v * math.cos(th) * dt
            F[1, 3] = math.sin(th) * dt
            F[1, 4] = 0.5 * v * (dt**2) * math.cos(th)
            F[2, 4] = dt

        G = np.array([
            [0.5 * dt**2 * math.cos(th), 0.0],
            [0.5 * dt**2 * math.sin(th), 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt]
        ], dtype=float)
        Q_noise = np.diag([self.sigma_a**2, self.sigma_alpha**2])
        Q = G @ Q_noise @ G.T

        self.P = F @ self.P @ F.T + Q
        self.time_since_last_measurement += dt
        return self.x.copy()

    def handle_occlusion(self) -> None:
        if not self._is_initialized:
            return
        self.P += self.Q_occ * self.dt

    def update(self, z_meas, ego_pose) -> tuple[bool, float]:
        if not self._is_initialized:
            xp, yp, th_p = ego_pose
            r_meas, phi_meas = z_meas
            abs_angle = th_p + phi_meas
            x0 = xp + r_meas * math.cos(abs_angle)
            y0 = yp + r_meas * math.sin(abs_angle)
            self.init_state(x0, y0, theta=abs_angle, v=0.0, omega=0.0)
            return True, 0.0

        xp, yp, th_p = ego_pose
        r_meas, phi_meas = float(z_meas[0]), float(z_meas[1])

        dx = self.x[0] - xp
        dy = self.x[1] - yp
        r_pred = math.sqrt(dx * dx + dy * dy)
        phi_pred = wrap_angle(math.atan2(dy, dx) - th_p)

        y_innov = np.array([r_meas - r_pred, wrap_angle(phi_meas - phi_pred)], dtype=float)

        # Measurement Jacobian H
        r2 = max(r_pred * r_pred, 1e-6)
        r = max(r_pred, 1e-3)
        H = np.zeros((2, 5), dtype=float)
        H[0, 0] = dx / r
        H[0, 1] = dy / r
        H[1, 0] = -dy / r2
        H[1, 1] = dx / r2

        S = H @ self.P @ H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, 999.0

        d_m2 = float(y_innov.T @ S_inv @ y_innov)
        if d_m2 > 5.991:
            return False, d_m2

        K = self.P @ H.T @ S_inv
        self.x += K @ y_innov
        self.x[2] = wrap_angle(float(self.x[2]))
        I = np.eye(5, dtype=float)
        self.P = (I - K @ H) @ self.P
        self.time_since_last_measurement = 0.0
        return True, d_m2

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x[:2].copy()

    def get_velocity_vector(self) -> np.ndarray:
        v = float(self.x[3])
        th = float(self.x[2])
        return np.array([v * math.cos(th), v * math.sin(th)], dtype=float)

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def is_confident(self) -> bool:
        if not self._is_initialized:
            return False
        pos_var = self.P[0, 0] + self.P[1, 1]
        if pos_var < 0:
            return False
        return bool(math.sqrt(pos_var) < self.confidence_pos_std_thresh and self.time_since_last_measurement < 3.0)
