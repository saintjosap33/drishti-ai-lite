import airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import csv
import os

class AirSimSurveillanceDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimSurveillanceDroneEnv, self).__init__()

        # Surveillance area config
        self.area_min = np.array([-50, -50, -30])
        self.area_max = np.array([50, 50, -10])
        self.grid_resolution = 10  # meters
        self.grid_points = self._generate_grid_points()
        self.visited = set()
        self.obstacles = set()

        # Drone step settings
        self.step_size = 10
        self.max_steps = 500
        self.current_position = np.array([0.0, 0.0, -10.0], dtype=np.float32)

        # Action space: Δx, Δy, Δz clipped between -1 and 1, scaled by step_size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: position + 6 sensor readings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.log_file = "training_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "step", "reward", "total_reward", "visited_count", "coverage_ratio", "collisions"])

        self.step_counter = 0
        self.episode_reward = 0.0
        self.episode_num = 0

        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self._initialize_drone()
        self.collisions = 0
        # Safe distance threshold for obstacle detection
        self.safe_distance = 2.0

    def _generate_grid_points(self):
        x = np.arange(self.area_min[0], self.area_max[0] + 1, self.grid_resolution)
        y = np.arange(self.area_min[1], self.area_max[1] + 1, self.grid_resolution)
        z = np.arange(self.area_min[2], self.area_max[2] + 1, self.grid_resolution)
        return set((int(i), int(j), int(k)) for i in x for j in y for k in z)

    def _initialize_drone(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(float(self.current_position[0]),
                                        float(self.current_position[1]),
                                        float(self.current_position[2]), velocity=5).join()

    def _get_observation(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

        distances = []
        sensor_names = ["DistanceFront", "DistanceBack", "DistanceLeft",
                        "DistanceRight", "DistanceUp", "DistanceDown"]
        for sensor in sensor_names:
            try:
                data = self.client.getDistanceSensorData(sensor)
                distances.append(data.distance if data.distance > 0 else 10.0)
            except:
                distances.append(10.0)  # fallback safe value

        return np.concatenate((position, np.array(distances))).astype(np.float32)

    def _position_to_grid(self, pos):
        return tuple((np.round(pos / self.grid_resolution) * self.grid_resolution).astype(int))

    def _is_obstacle_near_target(self):
        # Checks if any distance sensor reading is below the safe distance threshold
        sensor_names = ["DistanceFront", "DistanceBack", "DistanceLeft",
                        "DistanceRight", "DistanceUp", "DistanceDown"]

        for sensor in sensor_names:
            try:
                data = self.client.getDistanceSensorData(sensor)
                if 0 < data.distance < self.safe_distance:
                    return True
            except:
                continue  # Fail-safe: skip sensor if something goes wrong
        return False
    
    def seed(self, seed=None):
        import random

        if seed is None:
            seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        self._seed = seed
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Pass seed to parent, important for reproducibility
        if seed is not None:
            self.seed(seed)  # Set random seeds in your env too

        self._initialize_drone()
        self.current_position = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        self.step_counter = 0
        self.episode_reward = 0.0
        self.collisions = 0
        self.visited = set()
        self.episode_num += 1

        print(f"🚀 Starting Episode {self.episode_num}")

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.step_counter += 1
        done = False
        reward = 0.0

        # Normalize action and apply step size
        action = np.clip(action, -1, 1) * self.step_size
        target_pos = self.current_position + action
        target_pos = np.clip(target_pos, self.area_min, self.area_max)
        target_grid = self._position_to_grid(target_pos)

        # Obstacle handling
        if target_grid in self.obstacles:
            reward = 0.1
            print(f"🚫 Skipped blocked position {target_grid}")
        elif self._is_obstacle_near_target():
            reward = 0.1
            self.obstacles.add(target_grid)
            print(f"🚫 Obstacle detected near {target_grid}, skipping move")
        else:
            self.client.moveToPositionAsync(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), velocity=5).join()
            collision_info = self.client.simGetCollisionInfo()
            if collision_info.has_collided:
                reward = -10.0
                done = True
                self.collisions += 1
                self.obstacles.add(target_grid)
                print(f"💥 Collision at {target_grid}, ending episode")
            else:
                if target_grid not in self.visited:
                    reward = 1.0
                    self.visited.add(target_grid)
                    print(f"✅ Visited {target_grid}, total visited: {len(self.visited)}")
                else:
                    reward = -0.1  # Discourage revisits
                self.current_position = target_pos

        # Safe coverage calculation
        total_cells = max(1, len(self.grid_points) - len(self.obstacles))
        coverage = len(self.visited) / total_cells

        # Full coverage bonus
        if coverage >= 1.0:
            reward += 100
            done = True
            print("🎉 Full coverage achieved!")

        # Max steps termination
        if self.step_counter >= self.max_steps:
            truncated = True
            done = False
            print(f"⏰ Max steps {self.max_steps} reached, ending episode")
        else:
            truncated = False

        self.episode_reward += reward

        # Log progress every step
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.episode_num, self.step_counter, reward, self.episode_reward, len(self.visited), coverage, self.collisions])

        obs = self._get_observation()
        info = {
            "coverage": coverage,
            "collisions": self.collisions,
            "visited_count": len(self.visited),
            "step": self.step_counter,
            "episode_reward": self.episode_reward
        }

        return obs, reward, done, False, info

if __name__ == "__main__":
    env = AirSimSurveillanceDroneEnv()
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            # Random action for testing, replace with your RL agent's action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        print(f"Episode {episode + 1} finished. Total reward: {env.episode_reward}")
