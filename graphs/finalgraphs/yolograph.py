import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Setup ===
csv_path = r"D:\RL Navigation Drone\logs\yolov8\finalyolo.csv"  # Replace with your file path
save_dir = r"D:\RL Navigation Drone\graphs"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)

# === Plot 1: Loss curves ===
plt.figure(figsize=(10, 6))
for col, label, color in [
    ("box_loss", "Box Loss", "#F72585"),
    ("cls_loss", "Class Loss", "#3A0CA3"),
    ("dfl_loss", "Distribution Focal Loss", "#4895EF"),
]:
    sns.lineplot(x=df["Epoch"], y=df[col], label=label, linewidth=2.5, color=color)
plt.title("YOLOv8n Loss Components Over Epochs", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "yolo_loss_curves.png"), dpi=500)
plt.close()

# === Plot 2: Precision & Recall ===
plt.figure(figsize=(10, 6))
sns.lineplot(x=df["Epoch"], y=df["P"], label="Precision", color="#80ED99", linewidth=2.5)
sns.lineplot(x=df["Epoch"], y=df["R"], label="Recall", color="#FFB703", linewidth=2.5)
plt.title("Precision vs Recall Over Epochs", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "yolo_precision_recall.png"), dpi=500)
plt.close()

# === Plot 3: mAP50 and mAP50-95 ===
plt.figure(figsize=(10, 6))
sns.lineplot(x=df["Epoch"], y=df["mAP50"], label="mAP@0.5", color="#7209B7", linewidth=2.5)
sns.lineplot(x=df["Epoch"], y=df["mAP50-95"], label="mAP@0.5:0.95", color="#4CC9F0", linewidth=2.5)
plt.title("mAP Performance Over Epochs", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("mAP Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "yolo_map.png"), dpi=500)
plt.close()

# === Plot 4: GPU Usage ===
plt.figure(figsize=(10, 6))
sns.lineplot(x=df["Epoch"], y=df["GPU_mem"], color="#4361EE", linewidth=2.5)
plt.title("GPU Memory Usage Over Epochs", fontsize=16, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("GPU Memory (GB)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "yolo_gpu_mem.png"), dpi=500)
plt.close()

print(f"✅ All YOLOv8 training graphs saved to: {save_dir}")
