import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
base_path = r"D:\RL Navigation Drone\logs\rlnavi"
save_path = r"D:\RL Navigation Drone\graphs"
env_log_path = os.path.join(base_path, "patrolfixed.csv")

# === Load Data ===
env_log_df = pd.read_csv(env_log_path)

# === Aesthetic Setup ===
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# === Plot 1: Main PPO Training Metrics ===
main_metrics = [
    ("train/loss", "Training Loss", "#FF6B6B"),
    ("train/value_loss", "Value Loss", "#6A4C93"),
    ("rollout/ep_rew_mean", "Avg. Episode Reward", "#F9C80E"),
    ("rollout/ep_len_mean", "Avg. Episode Length", "#43AA8B"),
]

plt.figure(figsize=(12, 6))
for y_col, label, color in main_metrics:
    if y_col in env_log_df.columns:
        df = env_log_df[["time/total_timesteps", y_col]].dropna()
        df[y_col] = df[y_col].rolling(window=5, min_periods=1).mean()
        sns.lineplot(x=df["time/total_timesteps"], y=df[y_col], label=label, linewidth=2.5, color=color)

plt.title("Main PPO Training Metrics Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Timesteps")
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "ppo_main_metrics.png"), dpi=400)
plt.close()

# === Plot 2: Entropy Loss ===
if "train/entropy_loss" in env_log_df.columns:
    df_entropy = env_log_df[["time/total_timesteps", "train/entropy_loss"]].dropna()
    df_entropy["train/entropy_loss"] = df_entropy["train/entropy_loss"].rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=df_entropy["time/total_timesteps"],
        y=df_entropy["train/entropy_loss"],
        label="Entropy Loss",
        color="#F72585",
        linewidth=2.5
    )
    plt.title("Entropy Loss Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps")
    plt.ylabel("Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ppo_entropy_loss.png"), dpi=400)
    plt.close()

# === Plot 3: Explained Variance ===
if "train/explained_variance" in env_log_df.columns:
    df_var = env_log_df[["time/total_timesteps", "train/explained_variance"]].dropna()
    df_var["train/explained_variance"] = df_var["train/explained_variance"].rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=df_var["time/total_timesteps"],
        y=df_var["train/explained_variance"],
        label="Explained Variance",
        color="#3AFF82",
        linewidth=2.5
    )
    plt.title("Explained Variance Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps")
    plt.ylabel("Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ppo_explained_variance.png"), dpi=400)
    plt.close()

print("✅ All PPO training graphs saved to:", save_path)
