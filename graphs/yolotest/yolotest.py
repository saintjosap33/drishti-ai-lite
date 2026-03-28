import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
base_path = r"D:\RL Navigation Drone\logs"
save_path = r"D:\RL Navigation Drone\graphs"
csv_path = os.path.join(base_path, "yolo_test_predictions.csv")

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Aesthetic Setup ===
plt.style.use("dark_background")
sns.set_context("notebook", font_scale=1.25)

# === 1. Detections per Class ===
plt.figure(figsize=(10, 6))
class_counts = df['class'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel", edgecolor="black")
plt.title("Detections per Class", fontsize=18, color="#F72585", fontweight='bold')
plt.xlabel("Class", fontsize=14, color='white')
plt.ylabel("Count", fontsize=14, color='white')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "10_yolo_detections_per_class.png"), dpi=400, bbox_inches='tight', facecolor='black')
plt.close()

# === 2. Average Confidence per Class ===
plt.figure(figsize=(10, 6))
avg_conf = df.groupby("class")["confidence"].mean().sort_values(ascending=False)
sns.barplot(x=avg_conf.index, y=avg_conf.values, palette="coolwarm", edgecolor="black")
plt.title("Average Confidence per Class", fontsize=18, color="#58FCEC", fontweight='bold')
plt.xlabel("Class", fontsize=14, color='white')
plt.ylabel("Average Confidence", fontsize=14, color='white')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "11_yolo_avg_confidence_per_class.png"), dpi=400, bbox_inches='tight', facecolor='black')
plt.close()

# === 3. Low Confidence Predictions (< 0.5) ===
low_conf_df = df[df["confidence"] < 0.5]
low_conf_path = os.path.join(save_path, "12_yolo_low_confidence_predictions.csv")
low_conf_df.to_csv(low_conf_path, index=False)
print(f"⚠️ Low confidence predictions: {len(low_conf_df)} saved to {low_conf_path}")

# === Final Message ===
print("📈 YOLOv8 prediction graphs saved to:")
print(save_path)
