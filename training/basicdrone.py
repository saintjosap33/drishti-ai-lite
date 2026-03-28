from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv

class DroneNavigationEnv(gym.Env):
    def __init__(self, mode="basic", grid_size=(15, 15, 5), max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(6)  # 6 directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.goal_pos = np.array([5, 5, 2])
        self.start_pos = np.array([2, 2, 1])
        self.obstacles = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.step_count = 0
        self.total_reward = 0.0
        self.done_reason = 0
        return self._get_obs(), {}

    def _get_obs(self):
        directions = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
        obs = []
        for d in directions:
            new_pos = self.current_pos + d
            valid = all(0 <= new_pos[i] < self.grid_size[i] for i in range(3))
            blocked = tuple(new_pos) in self.obstacles if valid else True
            obs.append(1.0 if valid and not blocked else 0.0)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        prev_dist = np.linalg.norm(self.goal_pos - self.current_pos)

        move_vec = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1])
        }[int(action)]

        next_pos = self.current_pos + move_vec
        reward = -0.05  # small step penalty
        done = False

        if all(0 <= next_pos[i] < self.grid_size[i] for i in range(3)) and tuple(next_pos) not in self.obstacles:
            self.current_pos = next_pos
            new_dist = np.linalg.norm(self.goal_pos - self.current_pos)

            # Exponential proximity reward
            proximity_reward = 10 * np.exp(-0.8 * new_dist)
            improvement_bonus = max(0.0, prev_dist - new_dist) * 2.0
            reward += proximity_reward + improvement_bonus

            if new_dist < 1.0:
                reward += 100.0 + (self.max_steps - self.step_count) * 0.5
                done = True
                self.done_reason = 1
        else:
            reward = -10.0  # penalty for hitting wall/obstacle
            done = True
            self.done_reason = -1

        self.total_reward += reward
        if self.step_count >= self.max_steps:
            done = True
            self.done_reason = 0

        return self._get_obs(), reward, done, False, {
            "episode": {"r": self.total_reward, "l": self.step_count}
        }

    def render(self):
        print(f"Step: {self.step_count}, Pos: {tuple(self.current_pos)}, Goal: {tuple(self.goal_pos)}")

# === TRAINING ===

env = DroneNavigationEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=0)

os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "basic_ppo_log.csv")
with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'TotalReward', 'Steps', 'DoneReason'])

    obs, _ = env.reset()
    total_reward, step_count, episode = 0, 0, 0

    for step in range(40000):  # total training steps
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step_count += 1

        if done:
            print(f"[BASIC] Ep {episode} | Reward: {round(total_reward,2)} | Steps: {step_count} | Done: {env.done_reason}")
            writer.writerow([episode, round(total_reward, 2), step_count, env.done_reason])
            obs, _ = env.reset()
            total_reward, step_count = 0, 0
            episode += 1

model.save("basic_ppo_model")
print("✅ Saved BASIC PPO model")
