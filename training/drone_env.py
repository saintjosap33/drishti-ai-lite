# drone_env.py
import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import airsim
from stable_baselines3.common.callbacks import BaseCallback
import torch

class SurveillanceDroneEnv(gym.Env):
    def __init__(self, obstacle_csv="D:/RL Navigation Drone/data/obstacle_positions.csv"):
        super().__init__()

        self.area_min = np.array([-35, -117.5, 0])
        self.area_max = np.array([115, 112.5, 25])
        self.grid_res = 1

        self.grid = self._generate_grid_points()
        self.obstacles = set()
        self._load_obstacles_from_csv(obstacle_csv)

        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(6)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self._init_episode_vars()

    def _init_episode_vars(self):
        self.visited = set()
        self.global_visited = set()
        self.visited_points = {}
        self.prev_position = None
        self.current_idx = 0
        self.step_count = 0
        self.collisions = 0
        self.revisits = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.max_steps = 1500

    def _generate_grid_points(self):
        x = np.arange(self.area_min[0], self.area_max[0], self.grid_res)
        y = np.arange(self.area_min[1], self.area_max[1], self.grid_res)
        z = np.arange(self.area_min[2], self.area_max[2], self.grid_res)
        return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    def _load_obstacles_from_csv(self, file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            if str(row['Name']).lower() == 'ground':
                continue
            center = np.array([row['X'], row['Y'], row['Z']]) / 100.0
            dims = np.array([row['dimx'], row['dimy'], row['dimz']]) / 100.0
            min_corner = center - dims / 2
            max_corner = center + dims / 2
            for x in np.arange(min_corner[0], max_corner[0] + self.grid_res, self.grid_res):
                for y in np.arange(min_corner[1], max_corner[1] + self.grid_res, self.grid_res):
                    for z in np.arange(min_corner[2], max_corner[2] + self.grid_res, self.grid_res):
                        self.obstacles.add(tuple(np.round([x, y, z], 1)))

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._init_episode_vars()

        start = np.array([0.0, 0.0, 20.0])
        if not any((self.grid == start).all(axis=1)):
            self.grid = np.vstack([start, self.grid])
            self.current_idx = 0
        else:
            self.current_idx = np.where((self.grid == start).all(axis=1))[0][0]

        self._move_drone(self.grid[self.current_idx])
        return self.grid[self.current_idx], {}

    def step(self, action):
        self.step_count += 1
        self.total_steps += 1

        dx = np.array([[1, 0, 0], [-1, 0, 0],
                       [0, 1, 0], [0, -1, 0],
                       [0, 0, 1], [0, 0, -1]])

        next_pos = self.grid[self.current_idx] + dx[action]
        next_pos = np.array(next_pos).flatten()
        next_pos = np.round(next_pos, 1)
        reward = 0

        if tuple(next_pos) in self.obstacles or not self._is_valid(next_pos):
            self.collisions += 1
            reward = -1
        else:
            if tuple(self.grid[self.current_idx]) in self.visited:
                self.revisits += 1
            self.visited.add(tuple(self.grid[self.current_idx]))

            self.grid[self.current_idx] = next_pos
            self._move_drone(next_pos)
            reward = 1

        grid_pos = tuple(np.round(self.grid[self.current_idx], 1))
        self.visited_points[grid_pos] = self.visited_points.get(grid_pos, 0) + 1

        if self.prev_position is not None:
            dist = np.linalg.norm(self.grid[self.current_idx] - self.prev_position)
            if dist < self.grid_res * 0.5:
                reward -= 0.1

        revisit_ratio = sum(1 for v in self.visited_points.values() if v > 1) / max(len(self.visited_points), 1)
        reward += 0.5 * (1 - revisit_ratio)

        self.prev_position = self.grid[self.current_idx]
        self.episode_reward += reward

        done = self.step_count >= self.max_steps
        return self.grid[self.current_idx], reward, done, False, {}

    def _is_valid(self, pos):
        return np.all(pos >= self.area_min) and np.all(pos <= self.area_max) and pos[2] >= 1.0

    def _move_drone(self, pos, action=None, note=""):
    # Shift entire flight path upward by 20 meters
        x, y, z = float(pos[0]), float(pos[1]), float(-(pos[2] + 20.0))
        self.client.moveToPositionAsync(x, y, z, velocity=10, timeout_sec=3).join()


class CSVLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.logs = []
        self.start_time = time.time()

    def _on_step(self):
        if self.n_calls % self.model.n_steps == 0:
            vec_env = self.model.get_env()
            raw_env = vec_env.envs[0]
            valid_cells = max(len(raw_env.grid) - len(raw_env.obstacles), 1)
            area_coverage = min(len(raw_env.global_visited) / valid_cells, 1.0)
            path_efficiency = len(raw_env.global_visited) / max(raw_env.step_count, 1)
            obstacle_avoidance = 1 - raw_env.collisions / max(raw_env.step_count, 1)
            logs = {
                "timestep": self.num_timesteps,
                "area_coverage": round(area_coverage, 4),
                "path_efficiency": round(path_efficiency, 4),
                "obstacle_avoidance": round(obstacle_avoidance, 4),
                "collisions": raw_env.collisions,
                "revisits": raw_env.revisits,
                "avg_reward": round(raw_env.episode_reward / max(raw_env.step_count, 1), 4),
                "elapsed_time": round(time.time() - self.start_time, 2)
            }
            self.logs.append(logs)
        return True

    def _on_training_end(self):
        pd.DataFrame(self.logs).to_csv(self.log_path, index=False)
