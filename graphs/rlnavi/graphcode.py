import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
base_path = r"D:\RL Navigation Drone\logs"
save_path = r"D:\RL Navigation Drone\graphs"
env_log_path = os.path.join(base_path, "env log.csv")
agent_log_path = os.path.join(base_path, "agent log.csv")

# === Load CSVs ===
env_log_df = pd.read_csv(env_log_path)
agent_log_df = pd.read_csv(agent_log_path)

# === Graph Definitions ===
graph_info = [
    ("Training Loss over Time", "total_timesteps", "loss", env_log_df, "1_loss_over_time.png", "#F72585"),
    ("Entropy (Exploration) over Time", "total_timesteps", "entropy_loss", env_log_df, "2_entropy_over_time.png", "#58FCEC"), 
    ("Value Function Loss over Time", "total_timesteps", "value_loss", env_log_df, "3_value_loss_over_time.png", "#FFF95B"),  
    ("KL Divergence over Time", "total_timesteps", "approx_kl", env_log_df, "4_kl_divergence_over_time.png", "#4361EE"),
    ("Average Reward over Time", "timestep", "avg_reward", agent_log_df, "5_avg_reward_over_time.png", "#4CC9F0"),
    ("Obstacle Avoidance over Time", "timestep", "obstacle_avoidance", agent_log_df, "6_obstacle_avoidance.png", "#4895EF"),
    ("Collisions over Time", "timestep", "collisions", agent_log_df, "7_collisions_over_time.png", "#80FFDB"),
    ("Reward vs Episode Area Coverage", "episode_area_coverage", "avg_reward", agent_log_df, "8_reward_vs_coverage.png", "#64DFDF"),
    ("Path Efficiency over Time", "timestep", "path_efficiency", agent_log_df, "9_path_efficiency_over_time.png", "#FF9CEE"),
]

# === Aesthetic Setup ===
plt.style.use("dark_background")
sns.set_context("notebook", font_scale=1.25)

# === Plot and Save Each Graph ===
for title, x_col, y_col, df, filename, color in graph_info:
    plt.figure(figsize=(10, 6))
    
    # Plot with custom neon color
    sns.lineplot(x=df[x_col], y=df[y_col], color=color, linewidth=3)

    # Title and labels with glow aesthetic
    plt.title(title, fontsize=18, fontweight='bold', color=color)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=14, color='white')
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=14, color='white')
    
    # Style the grid and axes
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xticks(color='white', fontsize=11)
    plt.yticks(color='white', fontsize=11)
    sns.despine()

    # Save graph as high-res PNG
    full_path = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(full_path, dpi=400, bbox_inches='tight', facecolor='black')
    plt.close()

print("🖼️ Your NEON-STYLE DRONE GRAPHS are saved at:")
print(save_path)
