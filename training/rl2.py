import os
import csv
import argparse
from rl1 import LidarNavigationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, help="basic | obstacle | roundtrip")
parser.add_argument("--timesteps", type=int, required=True)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# === Paths ===
BASE_SAVE_DIR = r"D:\RL Navigation Drone\training\models\curriculum"
SAVE_DIR = os.path.join(BASE_SAVE_DIR, args.mode)
os.makedirs(SAVE_DIR, exist_ok=True)
model_path = os.path.join(SAVE_DIR, f"{args.mode}_model.zip")
vec_path = os.path.join(SAVE_DIR, f"{args.mode}_vecnormalize.pkl")
episode_csv = os.path.join(SAVE_DIR, f"{args.mode}_episodes.csv")

# === Custom Callback ===
class EpisodeSummaryCallback(BaseCallback):
    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode_num = 0
        self.file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Episode", "Steps", "Reward", "Result", "Global_Timesteps"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_num += 1
                steps = info["episode"]["l"]
                reward = round(info["episode"]["r"], 2)
                raw_env = self.training_env.envs[0].unwrapped
                mode = raw_env.mode
                done = "terminal_observation" in info
                roundtrip_done = getattr(raw_env, 'roundtrip_done', False)

                if mode == "roundtrip":
                    result = (
                        "✅ GOAL REACHED" if done and roundtrip_done and reward >= 200 else
                        "💥 COLLIDED" if reward < -5 else
                        "⌛ TIMEOUT"
                    )
                else:
                    result = (
                        "✅ GOAL REACHED" if done and reward >= 100 else
                        "💥 COLLIDED" if reward < -5 else
                        "⌛ TIMEOUT"
                    )

                self.writer.writerow([self.episode_num, steps, reward, result, self.num_timesteps])
                self.file.flush()
                print(f"📦 [{mode.upper()}] Ep {self.episode_num} | {result} | 🕹 {steps} steps | 🏆 {reward} | ⏱ {self.num_timesteps}")
        return True

    def _on_training_end(self) -> None:
        self.file.close()

# === Environment Setup ===
def make_env():
    max_steps = 500 if args.mode == "roundtrip" else 100
    return Monitor(LidarNavigationEnv(mode=args.mode, max_steps=max_steps))

env = DummyVecEnv([make_env])
vecnorm = VecNormalize(env, norm_obs=True, norm_reward=True)

# === Load or create model ===
if args.resume and os.path.exists(model_path):
    print(f"🔁 Resuming from checkpoint: {model_path}")
    model = PPO.load(model_path, env=vecnorm, device="cuda")
    vecnorm = VecNormalize.load(vec_path, env)
    reset_flag = False
else:
    model = PPO("MlpPolicy", vecnorm, verbose=0, device="cuda")
    reset_flag = True

# === Train ===
print(f"🚀 Training: {args.mode.upper()} | 🧠 Timesteps: {args.timesteps} | 🔁 Resume: {args.resume}")
model.learn(
    total_timesteps=args.timesteps,
    callback=EpisodeSummaryCallback(episode_csv),
    reset_num_timesteps=reset_flag
)

# === Save ===
model.save(model_path)
vecnorm.save(vec_path)
print(f"✅ Model saved to: {model_path}")
print(f"📄 Episode log saved to: {episode_csv}")
