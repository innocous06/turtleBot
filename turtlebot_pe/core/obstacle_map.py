import math
from typing import List, Dict, Tuple, Any
import numpy as np


class ObstacleMap:
    def __init__(
        self,
        resolution: float = 0.10,
        size: float = 12.0,
        hit_threshold: int = 3,
        arena_bounds: Tuple[float, float, float, float] = (-5.0, -5.0, 5.0, 5.0),
    ):
        self.resolution = float(resolution)
        self.size = float(size)
        self.hit_threshold = int(hit_threshold)
        self.arena_bounds = arena_bounds

        self.half_size = self.size / 2.0
        self.grid_dim = int(math.ceil(self.size / self.resolution))
        self.hit_counts = np.zeros((self.grid_dim, self.grid_dim), dtype=int)
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)

        self.obstacles: List[Dict[str, Any]] = []
        self._is_finalized = False

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((x + self.half_size) / self.resolution))
        gy = int(math.floor((y + self.half_size) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = (gx + 0.5) * self.resolution - self.half_size
        wy = (gy + 0.5) * self.resolution - self.half_size
        return wx, wy

    def is_in_grid(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim

    def add_scan(
        self,
        ranges: List[float],
        angle_min: float,
        angle_increment: float,
        ego_pose: Tuple[float, float, float],
        max_usable_range: float = 11.5,
    ) -> None:
        if self._is_finalized:
            return

        xp, yp, th_p = ego_pose
        arr_ranges = np.array(ranges, dtype=float)

        for i, r in enumerate(arr_ranges):
            if np.isnan(r) or np.isinf(r) or r <= 0.15 or r >= max_usable_range:
                continue

            angle = th_p + angle_min + i * angle_increment
            wx = xp + r * math.cos(angle)
            wy = yp + r * math.sin(angle)

            gx, gy = self.world_to_grid(wx, wy)
            if self.is_in_grid(gx, gy):
                self.hit_counts[gx, gy] += 1

    def finalize(self) -> None:
        self.grid = self.hit_counts >= self.hit_threshold
        self._is_finalized = True

        min_x, min_y, max_x, max_y = self.arena_bounds
        visited = np.zeros_like(self.grid, dtype=bool)
        self.obstacles = []

        # BFS clustering for static obstacles
        for gx in range(self.grid_dim):
            for gy in range(self.grid_dim):
                if self.grid[gx, gy] and not visited[gx, gy]:
                    component_cells = []
                    queue = [(gx, gy)]
                    visited[gx, gy] = True

                    while queue:
                        cx, cy = queue.pop(0)
                        component_cells.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                            nx, ny = cx + dx, cy + dy
                            if self.is_in_grid(nx, ny) and self.grid[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))

                    world_pts = np.array([self.grid_to_world(cx, cy) for cx, cy in component_cells])
                    if len(world_pts) < 4:
                        continue

                    center = np.mean(world_pts, axis=0)
                    cx, cy = float(center[0]), float(center[1])

                    # Ignore boundary walls
                    dist_to_wall = min(abs(cx - min_x), abs(cx - max_x), abs(cy - min_y), abs(cy - max_y))
                    if dist_to_wall < 0.35:
                        continue

                    dists = np.sqrt(np.sum((world_pts - center) ** 2, axis=-1))
                    radius = float(np.max(dists))
                    radius = max(0.20, min(1.20, radius))

                    self.obstacles.append({
                        "center": (cx, cy),
                        "radius": radius,
                        "points": world_pts,
                    })

    def get_obstacles(self) -> List[Dict[str, Any]]:
        return self.obstacles

    def get_arena_bounds(self) -> Tuple[float, float, float, float]:
        return self.arena_bounds

    def has_line_of_sight(self, from_xy: Tuple[float, float], to_xy: Tuple[float, float]) -> bool:
        # Bresenham raycast
        x0, y0 = self.world_to_grid(from_xy[0], from_xy[1])
        x1, y1 = self.world_to_grid(to_xy[0], to_xy[1])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        curr_x, curr_y = x0, y0

        while True:
            if (curr_x, curr_y) != (x0, y0) and (curr_x, curr_y) != (x1, y1):
                if self.is_in_grid(curr_x, curr_y):
                    if self.grid[curr_x, curr_y]:
                        return False

            if curr_x == x1 and curr_y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

        return True
