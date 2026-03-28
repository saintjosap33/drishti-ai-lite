import gymnasium as gym
import numpy as np

class LidarNavigationEnv(gym.Env):
    def __init__(self, mode="basic", max_steps=100):
        super().__init__()
        self.grid_size = (15, 15, 5)
        self.mode = mode
        self.max_steps = max_steps
        self.obstacles = set()

        self.start = None
        self.goal = None
        self.current = None
        self.step_count = 0
        self.total_reward = 0.0

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32)

        self._setup_scenario()
        self.reset()

    def _setup_scenario(self):
        self.obstacles.clear()
        self.start = np.array([0, 0, 0])

        if self.mode == "basic":
            self.goal = np.array([5, 0, 0])

        elif self.mode == "obstacle":
            self.goal = np.array([10, 5, 0])
            self.obstacles.update({
                (5, 2, 0), (6, 2, 0), (7, 3, 0), (8, 4, 0)
            })

        self.current = self.start.copy()

    def _in_bounds(self, pos):
        return np.all(pos >= 0) and np.all(pos < self.grid_size)

    def _get_lidar_readings(self, pos):
        dirs = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        readings = []
        for d in dirs:
            dist = 0.0
            test = pos.copy()
            for _ in range(10):
                test = test + d
                dist += 1.0
                if not self._in_bounds(test) or tuple(test) in self.obstacles:
                    break
            readings.append(dist)
        return np.array(readings, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._setup_scenario()
        self.step_count = 0
        self.total_reward = 0.0
        return self._get_lidar_readings(self.current), {}

    def step(self, action):
        self.step_count += 1
        move_vec = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])[action]
        next_pos = self.current + move_vec
        done = False
        reward = -0.1

        if not self._in_bounds(next_pos) or tuple(next_pos) in self.obstacles:
            reward = -10.0
            done = True
        else:
            self.current = next_pos
            dist = np.linalg.norm(self.goal - self.current)
            if dist < 1:
                reward = 100.0
                done = True
            else:
                reward = 1.0 / (dist + 1e-5)

        self.total_reward += reward
        if self.step_count >= self.max_steps:
            done = True

        return self._get_lidar_readings(self.current), reward, done, False, {
            "episode": {"r": self.total_reward, "l": self.step_count},
            "terminal_observation": done
        }
