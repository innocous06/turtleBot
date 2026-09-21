import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .math_utils import wrap_angle, clamp


class CatcherController:
    def __init__(
        self,
        v_max: float = 0.30,
        omega_max: float = 1.80,
        robot_radius: float = 0.18,
        arena_bounds: Tuple[float, float, float, float] = (-5.0, -5.0, 5.0, 5.0),
        pn_gain_N: float = 3.0,
        pn_bearing_gain: float = 1.8,
        pn_speed_coupling: float = 0.5,
        herd_offset: float = 0.75,
        herd_corner_offset: float = 1.50,
        corner_wall_threshold: float = 2.0,
        shadow_enter_distance: float = 0.70,
        shadow_exit_distance: float = 0.80,
        capture_radius: float = 0.50,
        capture_hold_target: float = 1.0,
        shadow_trail_distance: float = 0.35,
        shadow_Kx: float = 2.5,
        shadow_Ky: float = 4.0,
        shadow_Ktheta: float = 3.0,
        dwa_prediction_time: float = 1.5,
        dwa_safety_margin: float = 0.10,
        search_spin_rate: float = 1.0,
        search_timeout: float = 12.6,
        ekf_lost_timeout: float = 3.0,
    ):
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.robot_radius = float(robot_radius)
        self.arena_bounds = arena_bounds
        self.arena_center = np.array([
            (arena_bounds[0] + arena_bounds[2]) / 2.0,
            (arena_bounds[1] + arena_bounds[3]) / 2.0,
        ])

        self.pn_gain_N = float(pn_gain_N)
        self.pn_bearing_gain = float(pn_bearing_gain)
        self.pn_speed_coupling = float(pn_speed_coupling)
        self.herd_offset = float(herd_offset)
        self.herd_corner_offset = float(herd_corner_offset)
        self.corner_wall_threshold = float(corner_wall_threshold)

        self.shadow_enter_distance = float(shadow_enter_distance)
        self.shadow_exit_distance = float(shadow_exit_distance)
        self.capture_radius = float(capture_radius)
        self.capture_hold_target = float(capture_hold_target)
        self.shadow_trail_distance = float(shadow_trail_distance)
        self.shadow_Kx = float(shadow_Kx)
        self.shadow_Ky = float(shadow_Ky)
        self.shadow_Ktheta = float(shadow_Ktheta)

        self.dwa_prediction_time = float(dwa_prediction_time)
        self.dwa_safety_margin = float(dwa_safety_margin)

        self.search_spin_rate = float(search_spin_rate)
        self.search_timeout = float(search_timeout)
        self.ekf_lost_timeout = float(ekf_lost_timeout)

        self.state = "INIT"
        self.search_timer = 0.0
        self.capture_timer = 0.0
        self.capture_confirmed = False

    def set_state(self, new_state: str) -> None:
        self.state = new_state

    def get_state(self) -> str:
        return self.state

    def get_capture_timer(self) -> float:
        return self.capture_timer

    def is_capture_confirmed(self) -> bool:
        return self.capture_confirmed

    def _is_in_corner(self, x: float, y: float) -> bool:
        min_x, min_y, max_x, max_y = self.arena_bounds
        near_x = (abs(x - min_x) < self.corner_wall_threshold) or (abs(x - max_x) < self.corner_wall_threshold)
        near_y = (abs(y - min_y) < self.corner_wall_threshold) or (abs(y - max_y) < self.corner_wall_threshold)
        return near_x and near_y

    def _check_trajectory_collision(
        self,
        x0: float,
        y0: float,
        th0: float,
        v: float,
        omega: float,
        obstacles: List[Dict[str, Any]],
    ) -> bool:
        min_x, min_y, max_x, max_y = self.arena_bounds
        effective_radius = self.robot_radius + self.dwa_safety_margin
        steps = 10
        dt_step = self.dwa_prediction_time / steps
        curr_x, curr_y, curr_th = x0, y0, th0

        for _ in range(steps):
            if abs(omega) >= 1e-4:
                curr_th = wrap_angle(curr_th + omega * dt_step)
                curr_x += (v / omega) * (math.sin(curr_th) - math.sin(curr_th - omega * dt_step))
                curr_y += (v / omega) * (-math.cos(curr_th) + math.cos(curr_th - omega * dt_step))
            else:
                curr_x += v * math.cos(curr_th) * dt_step
                curr_y += v * math.sin(curr_th) * dt_step

            # Bounds check
            if (
                curr_x - effective_radius < min_x
                or curr_x + effective_radius > max_x
                or curr_y - effective_radius < min_y
                or curr_y + effective_radius > max_y
            ):
                return True

            # Obstacles check
            for obs in obstacles:
                ox, oy = obs["center"]
                ro = obs["radius"]
                if math.hypot(curr_x - ox, curr_y - oy) < (ro + effective_radius):
                    return True

        return False

    def compute_control(
        self,
        ego_pose: Tuple[float, float, float],
        ego_vel: Tuple[float, float],
        target_ekf_state: Optional[np.ndarray],
        obstacles: List[Dict[str, Any]],
        dt: float,
        bg_ready: bool = True,
        time_since_target_detected: float = 0.0,
    ) -> Tuple[float, float, str]:
        xp, yp, th_p = ego_pose
        vp, omegap = ego_vel

        # INIT
        if self.state == "INIT":
            if bg_ready:
                self.state = "SEARCH"
                self.search_timer = 0.0
            else:
                return 0.0, 0.0, self.state

        # CAPTURED
        if self.state == "CAPTURED":
            return 0.0, 0.0, self.state

        if target_ekf_state is None or time_since_target_detected > self.ekf_lost_timeout:
            self.state = "SEARCH"

        # SEARCH
        if self.state == "SEARCH":
            self.search_timer += dt
            if target_ekf_state is not None and time_since_target_detected < 0.5:
                self.state = "PURSUE"
                self.search_timer = 0.0
            else:
                if self.search_timer < self.search_timeout:
                    return 0.0, self.search_spin_rate, self.state
                spiral_v = clamp(0.12, 0.05, self.v_max)
                spiral_w = clamp(0.70, 0.20, self.omega_max)
                return spiral_v, spiral_w, self.state

        xe, ye, th_e, ve, om_e = target_ekf_state
        dx = xe - xp
        dy = ye - yp
        dist = math.hypot(dx, dy)

        if self.state == "PURSUE" and dist <= self.shadow_enter_distance:
            self.state = "SHADOW"
            self.capture_timer = 0.0
        elif self.state == "SHADOW" and dist > self.shadow_exit_distance:
            self.state = "PURSUE"
            self.capture_timer = 0.0

        # SHADOW
        if self.state == "SHADOW":
            if dist <= self.capture_radius:
                self.capture_timer += dt
                if self.capture_timer >= self.capture_hold_target:
                    self.capture_confirmed = True
                    self.state = "CAPTURED"
                    return 0.0, 0.0, self.state
            else:
                self.capture_timer = 0.0

            x_ref = xe - self.shadow_trail_distance * math.cos(th_e)
            y_ref = ye - self.shadow_trail_distance * math.sin(th_e)
            err_world = np.array([xp - x_ref, yp - y_ref], dtype=float)
            r_mat = np.array([
                [math.cos(th_e), math.sin(th_e)],
                [-math.sin(th_e), math.cos(th_e)],
            ])
            e_body = r_mat @ err_world
            e_x, e_y = float(e_body[0]), float(e_body[1])
            e_theta = wrap_angle(th_e - th_p)
            sinc_th = math.sin(e_theta) / e_theta if abs(e_theta) > 1e-4 else 1.0

            v_cmd = clamp(ve * math.cos(e_theta) - self.shadow_Kx * e_x, 0.05, self.v_max)
            w_cmd = clamp(
                om_e - self.shadow_Ky * ve * e_y * sinc_th + self.shadow_Ktheta * math.sin(e_theta),
                -self.omega_max,
                self.omega_max,
            )

            if not self._check_trajectory_collision(xp, yp, th_p, v_cmd, w_cmd, obstacles):
                return v_cmd, w_cmd, self.state
            return 0.10, clamp(w_cmd, -self.omega_max, self.omega_max), self.state

        # PURSUE
        to_center = self.arena_center - np.array([xe, ye])
        to_center_norm = to_center / max(np.linalg.norm(to_center), 1e-3)
        herd_d = self.herd_corner_offset if self._is_in_corner(xe, ye) else self.herd_offset
        p_herd = np.array([xe, ye]) + herd_d * to_center_norm
        target_los = math.atan2(p_herd[1] - yp, p_herd[0] - xp)

        vx_target = ve * math.cos(th_e)
        vy_target = ve * math.sin(th_e)
        vx_ego = vp * math.cos(th_p)
        vy_ego = vp * math.sin(th_p)

        vx_rel = vx_target - vx_ego
        vy_rel = vy_target - vy_ego
        los_rate = (dx * vy_rel - dy * vx_rel) / max(dist * dist, 1e-3)
        bearing_err = wrap_angle(target_los - th_p)

        w_raw = self.pn_gain_N * los_rate + self.pn_bearing_gain * bearing_err
        w_cmd = clamp(w_raw, -self.omega_max, self.omega_max)
        v_raw = self.v_max * max(0.0, math.cos(bearing_err)) / (1.0 + self.pn_speed_coupling * abs(w_cmd))
        v_cmd = clamp(v_raw, 0.0, self.v_max)

        # DWA check
        if not self._check_trajectory_collision(xp, yp, th_p, v_cmd, w_cmd, obstacles):
            return v_cmd, w_cmd, self.state

        best_cmd = (0.05, clamp(w_cmd * 1.5, -self.omega_max, self.omega_max))
        best_score = -1e9

        candidate_v = [self.v_max * 0.8, self.v_max * 0.5, 0.10, 0.05]
        candidate_w = [w_cmd, w_cmd + 0.6, w_cmd - 0.6, self.omega_max, -self.omega_max]

        for cv in candidate_v:
            for cw in candidate_w:
                cw = clamp(cw, -self.omega_max, self.omega_max)
                if not self._check_trajectory_collision(xp, yp, th_p, cv, cw, obstacles):
                    sim_th = wrap_angle(th_p + cw * self.dwa_prediction_time)
                    sim_x = xp + cv * math.cos(sim_th) * self.dwa_prediction_time
                    sim_y = yp + cv * math.sin(sim_th) * self.dwa_prediction_time
                    sim_dist = math.hypot(xe - sim_x, ye - sim_y)
                    score = -sim_dist + 0.5 * (cv / self.v_max)
                    if score > best_score:
                        best_score = score
                        best_cmd = (cv, cw)

        return best_cmd[0], best_cmd[1], self.state
