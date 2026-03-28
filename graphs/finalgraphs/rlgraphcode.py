import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
base_path = r"D:\RL Navigation Drone\logs\rlnavi"
save_path = r"D:\RL Navigation Drone\graphs"
os.makedirs(save_path, exist_ok=True)

# === Load CSVs ===
ppo_eval_df = pd.read_csv(os.path.join(base_path, "ppo_metrics.csv"))
a2c_eval_df = pd.read_csv(os.path.join(base_path, "a2c_metrics.csv"))

# === Success Rate Calculation ===
ppo_success = round(ppo_eval_df["Success"].mean() * 100, 2)
a2c_success = round(a2c_eval_df["Success"].mean() * 100, 2)

models = ["PPO", "A2C"]
success_rates = [ppo_success, a2c_success]
colors = ["#4895EF", "#9D4EDD"]

# === Plot Styling ===
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)

plt.figure(figsize=(8, 6))
barplot = sns.barplot(x=models, y=success_rates, palette=colors, edgecolor='black', linewidth=1.5)

# === Annotate Success % on Bars ===
for idx, val in enumerate(success_rates):
    barplot.text(idx, val + 2, f"{val:.2f}%", ha='center', va='bottom', fontsize=13, fontweight='bold')

# === Title & Axis Labels ===
plt.title("Final Success Rate of PPO vs A2C", fontsize=18, fontweight='bold')
plt.ylabel("Success Rate (%)", fontsize=14)
plt.xlabel("")  # No X-axis label needed
plt.ylim(0, 110)

# === Tweak Axis ===
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=12)
sns.despine(top=True, right=True, left=True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# === Save Figure ===
plt.tight_layout()
plt.savefig(os.path.join(save_path, "10_success_comparison.png"), dpi=500, bbox_inches='tight')
plt.close()

print(f"✅ Enhanced graph saved to: {save_path}\\10_success_comparison.png")
