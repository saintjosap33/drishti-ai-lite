from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env import SurveillanceDroneEnv, CSVLogger  # 🔥 import from your drone_env.py

def train_drone():
    env = DummyVecEnv([lambda: SurveillanceDroneEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env.training = True

    # ✅ Correctly unwrapped environment
    raw_env = env.venv.envs[0]
    raw_env.global_visited.clear()

    model = PPO("MlpPolicy", env, verbose=1)
    logger = CSVLogger("/content/surveillance_project/training_log.csv")
    model.learn(total_timesteps=100_000, callback=logger)

    model.save("/content/surveillance_project/surveillance_drone_model")
    env.save("/content/surveillance_project/vec_normalize.pkl")

    print("✅ Model saved as 'surveillance_drone_model.zip'")
    print("📄 Logs saved to 'training_log.csv'")
    print("🔹 VecNormalize stats saved to 'vec_normalize.pkl'")



if __name__ == "__main__":
    train_drone()
