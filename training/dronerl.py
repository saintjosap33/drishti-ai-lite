import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class DroneNavigationEnv(gym.Env):
    def __init__(self, mode="basic", grid_size=(15, 15, 5), max_steps=200):
        super(DroneNavigationEnv, self).__init__()
        self.mode = mode
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.goal_pos = np.array([10, 10, 2])
        self.start_pos = np.array([1, 1, 1])
        self.obstacles = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stuck_steps = 0
        self.done_reason = 0  # 0 = timeout, 1 = goal, 2 = stuck
        self.current_pos = self.start_pos.copy()
        self.reached_goal = False
        self.obstacles.clear()

        if self.mode in ['obstacle', 'roundtrip']:
            self.obstacles.add((5, 5, 1))  # One static obstacle

        return self._get_obs(), {}

    def _get_obs(self):
        directions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        obs = []
        for d in directions:
            new_pos = self.current_pos + d
            valid = all(0 <= new_pos[i] < self.grid_size[i] for i in range(3))
            blocked = tuple(new_pos) in self.obstacles if valid else True
            obs.append(1.0 if valid and not blocked else 0.0)
        return np.array(obs, dtype=np.float32)

    def _move(self, action):
        action = int(action)
        delta = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1])
        }[action]
        new_pos = self.current_pos + delta
        if all(0 <= new_pos[i] < self.grid_size[i] for i in range(3)) and tuple(new_pos) not in self.obstacles:
            self.current_pos = new_pos

    def step(self, action):
        self.steps += 1
        prev_pos = self.current_pos.copy()
        prev_dist = np.linalg.norm(prev_pos - self.goal_pos)

        self._move(action)
        new_dist = np.linalg.norm(self.current_pos - self.goal_pos)
        moved = not np.array_equal(prev_pos, self.current_pos)

        reward = -0.2  # step penalty
        done = False

        if moved:
            if new_dist < prev_dist:
                reward += 1.0  # strong progress reward
            elif new_dist > prev_dist:
                reward -= 0.5  # minor punishment
            self.stuck_steps = 0
        else:
            reward -= 2.0  # collision = pain
            self.stuck_steps += 1

        # Goal hit
        if not self.reached_goal and np.array_equal(self.current_pos, self.goal_pos):
            reward += 25
            self.reached_goal = True
            if self.mode != "roundtrip":
                self.done_reason = 1
                done = True

        # Return to start in roundtrip
        elif self.mode == "roundtrip" and self.reached_goal and np.array_equal(self.current_pos, self.start_pos):
            reward += 35
            self.done_reason = 1
            done = True

        # Stuck
        if self.stuck_steps >= 25:
            self.done_reason = 2
            done = True

        # Timeout
        if self.steps >= self.max_steps:
            self.done_reason = 0
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.steps}, Pos: {tuple(self.current_pos)}, Goal: {tuple(self.goal_pos)}")


def train_mode(mode_name, timesteps=100_000):
    env = DroneNavigationEnv(mode=mode_name)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=0)

    log_file = f"{mode_name}_ppo_log.csv"
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    done_types = {0: "⏳ Timeout", 1: "✅ Goal", 2: "🪤 Stuck"}

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'TotalReward', 'Steps', 'DoneReason'])

        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        episode = 0

        for _ in range(timesteps):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            step_count += 1

            if done:
                writer.writerow([episode, round(total_reward, 2), step_count, env.done_reason])
                print(f"[{mode_name.upper()}] Ep {episode:3} | Reward: {round(total_reward, 2):>6} | Steps: {step_count:<3} | Done: {env.done_reason} {done_types[env.done_reason]}")
                obs, _ = env.reset()
                total_reward = 0
                step_count = 0
                episode += 1

        model.save(f"{mode_name}_ppo_model")
        print(f"✅ Saved model for mode: {mode_name}")


if __name__ == "__main__":
    for mode in ['basic', 'obstacle', 'roundtrip']:
        print(f"\n=== 🚀 Training Mode: {mode.upper()} ===")
        train_mode(mode, timesteps=100_000)
