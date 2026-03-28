import gymnasium as gym
import numpy as np
import csv
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# === CALLBACK FOR TRAINING LOGGING ===
class TrainLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_num = 0
        with open(self.log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Steps", "Success", "Collisions"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        if len(dones) > 0 and dones[0]:
            self.episode_num += 1
            env = self.training_env.envs[0].unwrapped
            total_reward = getattr(env, "episode_reward", sum(rewards))
            total_steps = env.step_count
            success = int(env.success)
            collisions = env.collision_count
            with open(self.log_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_num, round(total_reward, 2), total_steps, success, collisions])
        return True

# === ENVIRONMENT ===
class ReachGoalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = (10, 10, 3)
        self.max_steps = 50
        self.obstacles = set()
        self.start_base = np.array([0.0, 0.0, 0.0])

        # Obstacle maze
        self.obstacles.update({(7, y, 1) for y in range(0, 4) if y != 2})
        self.obstacles.update({(8, 1, 1), (8, 3, 1), (8, 2, 0)})
        self.obstacles.update({(6, 2, 1), (7, 2, 0)})
        self.obstacles.update({(5, 2, 1), (5, 2, 0)})
        self.obstacles.update({(2, 2, 0), (3, 3, 0), (4, 4, 0)})

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0.0, high=150.0, shape=(10,), dtype=np.float32)

        self.goal_options = [
            np.array([7.0, 7.0, 1.0]),
            np.array([9.0, 4.0, 1.0])
        ]
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
                if not self._in_bounds(test) or tuple(test) in self.obstacles:
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

        # Select goal and store its index
        self.goal_index = random.randint(0, len(self.goal_options) - 1)
        self.goal = self.goal_options[self.goal_index]

        self.start = self.start_base.copy()
        self.current = self.start.copy()
        self.prev_dist = np.linalg.norm(self.goal - self.current)
        self.max_dist = np.linalg.norm(self.goal - self.start)
        return self._get_obs(), {}


    def step(self, action):
        self.step_count += 1
        delta = self.action_to_delta(action)
        next_pos = self.current + delta
        done = False
        reward = -0.1
        success_bonus = 200.0

        if not self._in_bounds(next_pos) or tuple(next_pos) in self.obstacles:
            reward -= 5.0
            self.collision_count += 1
            done = True
        else:
            self.current = next_pos
            new_dist = np.linalg.norm(self.goal - self.current)
            improvement = self.prev_dist - new_dist
            normalized_progress = improvement / self.max_dist if self.max_dist != 0 else 0
            reward += normalized_progress * 15.0
                        # Progressive shaping
            # (Optional) Distance penalty
            reward -= 0.01 * new_dist

            self.prev_dist = new_dist
            if new_dist < 1.0:
                reward += success_bonus
                self.success = True
                done = True

        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def action_to_delta(self, action):
        return [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ][action]

# === TRAIN ===
env = ReachGoalEnv()
check_env(env, warn=True)
train_logger = TrainLoggerCallback(log_path="ppo_train_logs.csv")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=2e-5,
    n_steps=4096,
    batch_size=64,
    n_epochs=30,
    gamma=0.99,
    gae_lambda=0.97,
    ent_coef=0.005,
    clip_range=0.2,
    clip_range_vf=None,
    vf_coef=0.8,
    tensorboard_log="./ppo_logs/",
    verbose=1,
    policy_kwargs=dict(
        net_arch=[128, 128],
        ortho_init=True
    )
)

model.learn(total_timesteps=300_000, callback=train_logger)
model.save("ppo_trained_model")
print("✅ Training done!")

# === EVALUATE ===
log_file = "ppo_metrics.csv"
with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Goal_Index", "Reward", "Steps", "Success", "Collisions"])

n_eval_episodes = 200
for ep in range(1, n_eval_episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([ep, env.goal_index, round(total_reward, 2), steps, int(env.success), env.collision_count])
    print(f"📊 Eval {ep} | Goal #{env.goal_index+1} | Reward: {total_reward:.2f} | Steps: {steps} | Success: {env.success} | Collisions: {env.collision_count}")
