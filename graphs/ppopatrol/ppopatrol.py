import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
base_path = r"D:\RL Navigation Drone\logs\rlnavi"
save_path = r"D:\RL Navigation Drone\graphs"
env_log_path = os.path.join(base_path, "ppo_patrol_train.csv")

# === Load CSV ===
env_log_df = pd.read_csv(env_log_path)

# === Graph Definitions ===
graph_info = [
    ("Training Loss over Time", "total_timesteps", "loss", "1_loss_over_time.png", "#F72585"),
    ("Entropy (Exploration) over Time", "total_timesteps", "entropy_loss", "2_entropy_over_time.png", "#58FCEC"),
    ("Value Function Loss over Time", "total_timesteps", "value_loss", "3_value_loss_over_time.png", "#FFF95B"),
    ("KL Divergence over Time", "total_timesteps", "approx_kl", "4_kl_divergence_over_time.png", "#4361EE"),
    ("Explained Variance over Time", "total_timesteps", "explained_variance", "5_explained_variance_over_time.png", "#3AFF82"),
]

# === Aesthetic Setup ===
plt.style.use("dark_background")
sns.set_context("notebook", font_scale=1.25)

# === Plot and Save Each Graph ===
for title, x_col, y_col, filename, color in graph_info:
    plt.figure(figsize=(10, 6))

    sns.lineplot(x=env_log_df[x_col], y=env_log_df[y_col], color=color, linewidth=3)

    plt.title(title, fontsize=18, fontweight='bold', color=color)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=14, color='white')
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=14, color='white')
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xticks(color='white', fontsize=11)
    plt.yticks(color='white', fontsize=11)
    sns.despine()

    full_path = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(full_path, dpi=400, bbox_inches='tight', facecolor='black')
    plt.close()

print("✅ NEON-STYLE DRONE GRAPHS saved to:")
print(save_path)
