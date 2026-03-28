import gymnasium as gym
import numpy as np
import csv
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os

# === Load Obstacles from Real Map ===
def load_obstacle_set(csv_path, grid_res=50):
    df = pd.read_csv(csv_path)
    for col in ["X", "Y", "Z", "dimx", "dimy", "dimz"]:
        df[col] = df[col] / 100.0

    obstacles = set()
    for _, row in df.iterrows():
        x = int(row["X"] // grid_res)
        y = int(row["Y"] // grid_res)
        z = int(row["Z"] // grid_res)
        dx = int(row["dimx"] // grid_res)
        dy = int(row["dimy"] // grid_res)
        dz = int(row["dimz"] // grid_res)
        for i in range(dx + 1):
            for j in range(dy + 1):
                for k in range(dz + 1):
                    obstacles.add((x + i, y + j, z + k))
    return obstacles

class TrainLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_num = 0
        with open(self.log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Steps", "Success", "Collisions"])

    def _on_step(self) -> bool:
        # Use self.locals['infos'] to check for episode end
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_num += 1
                env = self.training_env.envs[0].unwrapped
                total_reward = info["episode"]["r"]
                total_steps = info["episode"]["l"]
                success = int(getattr(env, "success", 0))
                collisions = getattr(env, "collision_count", 0)
                with open(self.log_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.episode_num, round(total_reward, 2), total_steps, success, collisions])
        return True

class CSVLoggerCallback(BaseCallback):
    def __init__(self, csv_path="metrics_log.csv", verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.first_write = True

    def _on_step(self) -> bool:
        # Save logs only every rollout (so once per update)
        return True

    def _on_rollout_end(self) -> None:
        logs = self.model.logger.name_to_value  # Dict of all training logs
        metrics = {
            "iteration": logs.get("rollout/ep_rew_mean", 0),  # Used as progress marker
            "ep_len_mean": logs.get("rollout/ep_len_mean", 0),
            "ep_rew_mean": logs.get("rollout/ep_rew_mean", 0),
            "fps": logs.get("time/fps", 0),
            "time_elapsed": logs.get("time/time_elapsed", 0),
            "total_timesteps": logs.get("time/total_timesteps", 0),
            "approx_kl": logs.get("train/approx_kl", 0),
            "clip_fraction": logs.get("train/clip_fraction", 0),
            "clip_range": self.model.clip_range if callable(self.model.clip_range) == False else self.model.clip_range(1),
            "entropy_loss": logs.get("train/entropy_loss", 0),
            "explained_variance": logs.get("train/explained_variance", 0),
            "learning_rate": logs.get("train/learning_rate", 0),
            "loss": logs.get("train/loss", 0),
            "n_updates": logs.get("train/n_updates", 0),
            "policy_gradient_loss": logs.get("train/policy_gradient_loss", 0),
            "value_loss": logs.get("train/value_loss", 0),
        }

        # Write to CSV
        write_header = not os.path.exists(self.csv_path) or self.first_write
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
                self.first_write = False
            writer.writerow(metrics)


# === ENVIRONMENT ===
class PatrolEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = (25, 25, 5)
        self.max_steps = 400
        self.obstacles = load_obstacle_set("D:/RL Navigation Drone/data/obstacle_positions.csv")

        self.patrol_route = [
            np.array([6, 8, 1]),     # Start
            np.array([6, 14, 3]),    # GOAL 1: Lift vertically to 3
            np.array([11, 13, 2]),   # GOAL 2: Slight descend
            np.array([11, 8, 4]),    # GOAL 3: Climb again
            np.array([6, 8, 1])      # RETURN: Descend to starting Z
        ]


        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*6 + [-25]*3 + [0], dtype=np.float32),
            high=np.array([10]*6 + [25]*3 + [self.max_steps], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def _in_bounds(self, pos):
        return np.all(pos >= 0) and np.all(pos < self.grid_size)

    def _get_lidar_readings(self, pos):
        dirs = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                np.array([0, 1, 0]), np.array([0, -1, 0]),
                np.array([0, 0, 1]), np.array([0, 0, -1])]
        readings = []
        for d in dirs:
            test = pos.copy()
            dist = 0
            for _ in range(10):
                test += d
                dist += 1
                if not self._in_bounds(test) or tuple(test.astype(int)) in self.obstacles:
                    break
            readings.append(float(dist))
        return np.array(readings, dtype=np.float32)

    def _get_obs(self):
        lidar = self._get_lidar_readings(self.current)
        to_goal = self.goal - self.current
        steps_left = np.array([self.max_steps - self.step_count], dtype=np.float32)
        return np.concatenate([lidar, to_goal, steps_left]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.collision_count = 0
        self.success = False
        self.route_index = 1
        self.start = np.array([6, 8, 1])
        self.current = self.start.copy()
        self.goal = self.patrol_route[self.route_index]
        self.prev_dist = np.linalg.norm(self.goal - self.current)
        self.max_dist = self.prev_dist
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        delta = self._delta(action)
        next_pos = self.current + delta
        next_pos = np.clip(next_pos, [0, 0, 0], np.array(self.grid_size) - 1)

        done = False
        reward = -0.1  # movement cost

        # Small boost for Z-axis movement to prevent ignoring vertical
        if action in [4, 5]:
            reward += 0.2

        if tuple(next_pos.astype(int)) in self.obstacles:
            reward -= 5.0
            self.collision_count += 1
            done = True
        else:
            self.current = next_pos.astype(int)
            new_dist = np.linalg.norm(self.goal - self.current)
            improvement = self.prev_dist - new_dist
            normalized_progress = improvement / self.max_dist if self.max_dist != 0 else 0
            reward += normalized_progress * 15.0
            reward -= 0.01 * new_dist
            self.prev_dist = new_dist

            if new_dist < 1.0:
                reward += 10.0  # reached an intermediate goal
                self.route_index += 1

                if self.route_index >= len(self.patrol_route):
                    if np.linalg.norm(self.current - self.start) < 1.0:
                        reward += 250.0  # 🎯 full loop complete
                        self.success = True
                    done = True
                else:
                    self.goal = self.patrol_route[self.route_index]
                    self.prev_dist = np.linalg.norm(self.goal - self.current)
                    self.max_dist = self.prev_dist

        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _delta(self, action):
        return [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ][action]



# Put this EXACTLY at the bottom of your file 👇
if __name__ == "__main__":
    # === TRAIN ===
    env = PatrolEnv()
    check_env(env, warn=True)
    logger = TrainLoggerCallback(log_path="ppo_patrol_logs.csv")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=32,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.92,
        ent_coef=0.001,
        clip_range=0.1,
        vf_coef=1.0,
        tensorboard_log="./ppo_logs/",
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128], ortho_init=True)
    )

    csv_logger = CSVLoggerCallback(csv_path="ppo_training_metrics.csv")
    model.learn(total_timesteps=300_000, callback=logger)
    model.save("ppo_patrol_model")
    print("✅ Training complete.")

    # === EVALUATE ===
    print("\n🚁 Drone Patrol Path Execution (Post Training)")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1

        print(f"🔄 Step {step:03d} | Pos: {env.current} | Goal: {env.goal} | Reward: {round(reward,2)}")
        print(f"📡 LIDAR: {env._get_lidar_readings(env.current)}\n")

    print(f"🏁 Patrol Done in {step} steps | Final Reward: {round(total_reward, 2)} | Success: {env.success}")
