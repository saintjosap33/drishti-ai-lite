import sys
sys.path.append("D:/RL Navigation Drone/training")

import time
import airsim
import numpy as np
from stable_baselines3 import PPO
from ppocurriculum_trainer import PatrolEnv

# === Config ===
SCALE = 2.5  # scaling factor: 1 grid unit = 2.5 meters

# === AirSim Setup ===
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-SCALE, 2).join()  # stabilize at Z = -2.5 meters

# === Load RL Env + Model ===
env = PatrolEnv()
model = PPO.load("D:/RL Navigation Drone/training/ppo_patrol_model")

# === Evaluation + Live Drone Movement ===
print("\n🚁 Drone Patrol Path Execution (Clean Evaluation + AirSim)")
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    step += 1

    grid_pos = env.current
    x = float(grid_pos[0]) * SCALE
    y = float(grid_pos[1]) * SCALE
    z = -float(grid_pos[2]) * SCALE  # AirSim Z-axis is flipped

    print(f"🔄 Step {step:03d} | Pos: {env.current} | Goal: {env.goal} | Reward: {round(reward, 2)}")
    print(f"📡 LIDAR: {env._get_lidar_readings(env.current)}")
    print(f"🚀 Moving drone to: X={x}, Y={y}, Z={z} meters\n")

    client.moveToPositionAsync(x, y, z, velocity=3).join()
    time.sleep(0.4)

print(f"🏁 Patrol Done in {step} steps | Final Reward: {round(total_reward, 2)} | Success: {env.success}")

# === Shutdown ===
client.hoverAsync().join()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
