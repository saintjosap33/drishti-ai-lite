import airsim
import numpy as np
import time
import torch
import cv2
import pandas as pd
from ultralytics import YOLO
from stable_baselines3 import PPO
from gymnasium import spaces

# === Load YOLOv8 Model ===
yolo_model = YOLO("yolo_human_detection/weights/best.pt")  # path to your trained model

def detect_humans(image):
    results = yolo_model.predict(source=image, conf=0.4, verbose=False)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # human
                return True, box
    return False, None

# === RL Navigation Env Wrapper ===
class DroneEnv:
    def __init__(self, client):
        self.client = client
        self.grid_res = 50  # Must match the training
        self.grid_size = (25, 25, 5)
        self.action_space = spaces.Discrete(6)
        self.current_pos = np.array([6, 8, 1])

    def reset(self):
        self.current_pos = np.array([6, 8, 1])
        return self._get_obs()

    def _get_obs(self):
        # Request lidar data from AirSim
        lidar = self.client.getLidarData()
        if len(lidar.point_cloud) < 3:
            return np.zeros(9, dtype=np.float32)

        points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        # Simplified: take min dist in 6 directions
        dirs = [np.array([1,0,0]), np.array([-1,0,0]), np.array([0,1,0]),
                np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1])]
        dists = []
        for d in dirs:
            proj = points @ d
            dists.append(np.min(proj[proj > 0]) if np.any(proj > 0) else 10)

        goal = np.array([11, 8, 4])  # Temp static goal
        return np.concatenate([np.array(dists), goal - self.current_pos])

    def step(self, action):
        move = [np.array([1,0,0]), np.array([-1,0,0]),
                np.array([0,1,0]), np.array([0,-1,0]),
                np.array([0,0,1]), np.array([0,0,-1])][action]

        self.current_pos += move
        self.current_pos = np.clip(self.current_pos, [0,0,0], np.array(self.grid_size) - 1)
        x, y, z = self.current_pos * self.grid_res
        self.client.moveToPositionAsync(x, y, -z, 2).join()

        reward = -0.1
        done = np.linalg.norm(self.current_pos - np.array([11, 8, 4])) < 1.0
        return self._get_obs(), reward, done, {}

# === Drone + YOLO + RL Integration ===
def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    env = DroneEnv(client)
    model = PPO.load("ppo_patrol_model")
    obs = env.reset()
    human_log = []

    for step in range(300):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

        if step % 5 == 0:
            response = client.simGetImage("0", airsim.ImageType.Scene)
            if response:
                img1d = np.frombuffer(response, dtype=np.uint8)
                img_rgb = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
                found, box = detect_humans(img_rgb)
                if found:
                    pos = client.getMultirotorState().kinematics_estimated.position
                    human_log.append([pos.x_val, pos.y_val, pos.z_val])
                    print(f"👤 Human detected at: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")

        if done:
            print("✅ Goal Reached")
            break

    df = pd.DataFrame(human_log, columns=["x", "y", "z"])
    df.to_csv("detected_humans.csv", index=False)
    print("📁 Human detection log saved to detected_humans.csv")

if __name__ == "__main__":
    main()
