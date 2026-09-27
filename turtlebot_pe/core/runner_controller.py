import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .math_utils import wrap_angle, clamp


class RunnerController:
    def __init__(
        self,
        v_max: float = 0.30,
        omega_max: float = 1.80,
        robot_radius: float = 0.18,
        arena_bounds: Tuple[float, float, float, float] = (-5.0, -5.0, 5.0, 5.0),
        dt: float = 0.05,
        apf_eta_pursuer: float = 3.5,
        apf_d_pursuer_max: float = 4.0,
        apf_beta_closing: float = 1.5,
        apf_eta_obstacle: float = 2.0,
        apf_d_obstacle_max: float = 0.80,
        apf_k_vortex: float = 1.2,
        apf_eta_wall: float = 2.5,
        apf_d_wall_max: float = 1.0,
        apf_k_center: float = 0.08,
        heading_gain: float = 2.5,
        ou_tau: float = 0.8,
        ou_sigma: float = 0.5,
        ou_activation_distance: float = 2.0,
        shield_activation_distance: float = 2.0,
        shield_time_advantage: float = 0.5,
        shield_orbit_margin: float = 0.35,
        shield_exit_distance: float = 3.0,
        match_duration: float = 180.0,
        endgame_start: float = 150.0,
        endgame_distance_gain: float = 0.5,
        endgame_ou_sigma: float = 0.2,
    ):
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.robot_radius = float(robot_radius)
        self.arena_bounds = arena_bounds
        self.arena_center = np.array([
            (arena_bounds[0] + arena_bounds[2]) / 2.0,
            (arena_bounds[1] + arena_bounds[3]) / 2.0,
        ])
        self.dt = float(dt)

        # APF parameters
        self.eta_p = float(apf_eta_pursuer)
        self.d_p0 = float(apf_d_pursuer_max)
        self.beta_v = float(apf_beta_closing)
        self.eta_o = float(apf_eta_obstacle)
        self.d_o0 = float(apf_d_obstacle_max)
        self.k_vortex = float(apf_k_vortex)
        self.eta_w = float(apf_eta_wall)
        self.d_w0 = float(apf_d_wall_max)
        self.k_center = float(apf_k_center)
        self.k_theta = float(heading_gain)

        # Anti-prediction OU noise
        self.ou_tau = float(ou_tau)
        self.ou_sigma = float(ou_sigma)
        self.ou_activation_distance = float(ou_activation_distance)
        self.xi = 0.0

        # Shielding parameters
        self.shield_activation_distance = float(shield_activation_distance)
        self.shield_time_advantage = float(shield_time_advantage)
        self.shield_orbit_margin = float(shield_orbit_margin)
        self.shield_exit_distance = float(shield_exit_distance)
        self.active_shield_obs: Optional[Dict[str, Any]] = None

        # Timing & Endgame
        self.match_duration = float(match_duration)
        self.endgame_start = float(endgame_start)
        self.endgame_distance_gain = float(endgame_distance_gain)
        self.endgame_ou_sigma = float(endgame_ou_sigma)

        # FSM state
        self.state = "INIT"

    def set_state(self, new_state: str) -> None:
        self.state = new_state

    def get_state(self) -> str:
        return self.state

    def _update_ou_noise(self, dt: float, current_sigma: float) -> float:
        # Ornstein-Uhlenbeck process
        decay = 1.0 - (dt / self.ou_tau)
        noise = current_sigma * math.sqrt(dt) * float(np.random.normal(0.0, 1.0))
        self.xi = self.xi * max(0.0, decay) + noise
        return float(np.clip(self.xi, -math.pi / 3.0, math.pi / 3.0))

    def compute_control(
        self,
        ego_pose: Tuple[float, float, float],
        catcher_ekf_state: Optional[np.ndarray],
        obstacles: List[Dict[str, Any]],
        elapsed_time: float,
        dt: float,
        bg_ready: bool = True,
    ) -> Tuple[float, float, str]:
        xp, yp, th_p = ego_pose
        p_ego = np.array([xp, yp], dtype=float)

        # INIT
        if self.state == "INIT":
            if bg_ready:
                self.state = "EVADE"
            else:
                return 0.0, 0.0, self.state

        # SURVIVED
        if elapsed_time >= self.match_duration or self.state == "SURVIVED":
            self.state = "SURVIVED"
            return 0.0, 0.0, self.state

        if catcher_ekf_state is None:
            # Maintain position away from boundaries if catcher not yet tracked
            to_center = self.arena_center - p_ego
            center_dist = float(np.linalg.norm(to_center))
            if center_dist > 1.0:
                des_th = math.atan2(to_center[1], to_center[0])
                e_th = wrap_angle(des_th - th_p)
                w_cmd = clamp(self.k_theta * e_th, -self.omega_max, self.omega_max)
                v_cmd = clamp(self.v_max * 0.4 * max(0.0, math.cos(e_th)), 0.0, self.v_max)
                return v_cmd, w_cmd, self.state
            return 0.0, 0.0, self.state

        p_catcher = catcher_ekf_state[:2]
        th_c = float(catcher_ekf_state[2])
        v_c = float(catcher_ekf_state[3])
        v_catcher_vec = np.array([v_c * math.cos(th_c), v_c * math.sin(th_c)], dtype=float)

        d_p = float(np.linalg.norm(p_ego - p_catcher))
        d_p = max(d_p, 1e-3)

        # Check SHIELD transition
        if self.state == "EVADE" and d_p < self.shield_activation_distance:
            best_obs = None
            best_advantage = -999.0
            for obs in obstacles:
                c_obs = np.array(obs["center"], dtype=float)
                r_obs = float(obs["radius"])
                obs_to_catcher = p_catcher - c_obs
                d_c_obs = float(np.linalg.norm(obs_to_catcher))
                if d_c_obs < 1e-3:
                    continue
                dir_away = -obs_to_catcher / d_c_obs
                p_shadow = c_obs + (r_obs + self.shield_orbit_margin) * dir_away
                t_runner = float(np.linalg.norm(p_ego - p_shadow)) / self.v_max
                t_catcher = d_c_obs / self.v_max
                advantage = t_catcher - t_runner
                if advantage > self.shield_time_advantage and advantage > best_advantage:
                    best_advantage = advantage
                    best_obs = obs

            if best_obs is not None:
                self.state = "SHIELD"
                self.active_shield_obs = best_obs

        elif self.state == "SHIELD":
            if d_p > self.shield_exit_distance or self.active_shield_obs is None:
                self.state = "EVADE"
                self.active_shield_obs = None

        # State 2: SHIELD
        if self.state == "SHIELD" and self.active_shield_obs is not None:
            c_obs = np.array(self.active_shield_obs["center"], dtype=float)
            r_orbit = float(self.active_shield_obs["radius"]) + self.shield_orbit_margin
            phi_desired = math.atan2(p_catcher[1] - c_obs[1], p_catcher[0] - c_obs[0]) + math.pi
            p_orbit = c_obs + r_orbit * np.array([math.cos(phi_desired), math.sin(phi_desired)])
            target_vec = p_orbit - p_ego
            des_th = math.atan2(target_vec[1], target_vec[0])
            e_th = wrap_angle(des_th - th_p)
            w_cmd = clamp(self.k_theta * e_th, -self.omega_max, self.omega_max)
            v_cmd = clamp(self.v_max * max(0.0, math.cos(e_th)), 0.05, self.v_max)
            return v_cmd, w_cmd, self.state

        # State 1: EVADE (Artificial Potential Fields)
        # Force 1: Pursuer repulsion
        f_pur = np.zeros(2, dtype=float)
        if d_p <= self.d_p0:
            n_p = (p_ego - p_catcher) / d_p
            # Closing velocity calculation
            rel_vel = v_catcher_vec
            closing_vel = -float(np.dot(n_p, rel_vel))
            scaling = 1.0 + self.beta_v * max(0.0, closing_vel)
            rep_mag = self.eta_p * (1.0 / d_p - 1.0 / self.d_p0) * (1.0 / (d_p * d_p)) * scaling
            f_pur = rep_mag * n_p

        # Force 2: Obstacle repulsion + vortex
        f_obs_total = np.zeros(2, dtype=float)
        rot_j = np.array([[0.0, -1.0], [1.0, 0.0]])
        for obs in obstacles:
            c_obs = np.array(obs["center"], dtype=float)
            r_obs = float(obs["radius"])
            vec_to_ego = p_ego - c_obs
            d_center = float(np.linalg.norm(vec_to_ego))
            if d_center < 1e-3:
                continue
            d_surface = max(1e-3, d_center - r_obs)
            if d_surface <= self.d_o0:
                n_oi = vec_to_ego / d_center
                rep_mag = self.eta_o * (1.0 / d_surface - 1.0 / self.d_o0) * (1.0 / (d_surface * d_surface))
                f_rep = rep_mag * n_oi
                # Tangential vortex direction
                cross_val = (p_catcher[0] - p_ego[0]) * (c_obs[1] - p_ego[1]) - (p_catcher[1] - p_ego[1]) * (c_obs[0] - p_ego[0])
                s_i = 1.0 if cross_val >= 0.0 else -1.0
                f_vortex = s_i * self.k_vortex * float(np.linalg.norm(f_rep)) * (rot_j @ n_oi)
                f_obs_total += f_rep + f_vortex

        # Force 3: Wall repulsion
        min_x, min_y, max_x, max_y = self.arena_bounds
        f_wall = np.zeros(2, dtype=float)
        d_left = p_ego[0] - min_x
        if 0 < d_left < self.d_w0:
            f_wall[0] += self.eta_w * (1.0 / d_left - 1.0 / self.d_w0) * (1.0 / (d_left * d_left))
        d_right = max_x - p_ego[0]
        if 0 < d_right < self.d_w0:
            f_wall[0] -= self.eta_w * (1.0 / d_right - 1.0 / self.d_w0) * (1.0 / (d_right * d_right))
        d_bottom = p_ego[1] - min_y
        if 0 < d_bottom < self.d_w0:
            f_wall[1] += self.eta_w * (1.0 / d_bottom - 1.0 / self.d_w0) * (1.0 / (d_bottom * d_bottom))
        d_top = max_y - p_ego[1]
        if 0 < d_top < self.d_w0:
            f_wall[1] -= self.eta_w * (1.0 / d_top - 1.0 / self.d_w0) * (1.0 / (d_top * d_top))

        # Force 4: Center bias vs. Endgame max-distance waypoint
        current_ou_sigma = self.ou_sigma
        if elapsed_time >= self.endgame_start:
            # Diametrically opposite point in arena relative to catcher
            opposite_point = 2.0 * self.arena_center - p_catcher
            opposite_point[0] = np.clip(opposite_point[0], min_x + 1.0, max_x - 1.0)
            opposite_point[1] = np.clip(opposite_point[1], min_y + 1.0, max_y - 1.0)
            f_center = self.endgame_distance_gain * (opposite_point - p_ego)
            current_ou_sigma = self.endgame_ou_sigma
        else:
            f_center = -self.k_center * (p_ego - self.arena_center)

        f_net = f_pur + f_obs_total + f_wall + f_center
        if float(np.linalg.norm(f_net)) < 1e-3:
            return 0.0, 0.0, self.state

        # Anti-prediction heading perturbation
        noise_heading = 0.0
        if d_p < self.ou_activation_distance:
            noise_heading = self._update_ou_noise(dt, current_ou_sigma)

        theta_des = math.atan2(f_net[1], f_net[0]) + noise_heading
        e_th = wrap_angle(theta_des - th_p)
        w_cmd = clamp(self.k_theta * e_th, -self.omega_max, self.omega_max)
        v_cmd = clamp(self.v_max * max(0.0, math.cos(e_th)), 0.0, self.v_max)

        return v_cmd, w_cmd, self.state
