import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Set up paths ===
base_path = r"D:\RL Navigation Drone\logs"
save_path = r"D:\RL Navigation Drone\graphs"
classwise_path = os.path.join(base_path, "classwise_metrics.csv")
training_path = os.path.join(base_path, "training_results_complete.csv")

# === Load CSVs ===
class_df = pd.read_csv(classwise_path)
train_df = pd.read_csv(training_path)

# === Set dark theme and aesthetics ===
plt.style.use("dark_background")
sns.set_context("notebook", font_scale=1.2)

# === Custom graph config (title, x, y, color, filename) ===
graph_definitions = [
    # From training_results_complete.csv
    ("Training Box Loss over Epochs", "Epoch", "box_loss", "#F72585", train_df, "1_train_loss.png"),
    ("Validation Class Loss over Epochs", "Epoch", "cls_loss", "#3A0CA3", train_df, "2_val_loss.png"),
    ("mAP@0.5 and mAP@0.5:0.95 over Epochs", "Epoch", ["mAP50", "mAP50_95"], ["#4CC9F0", "#80FFDB"], train_df, "3_map_curves.png"),
    ("Box Precision & Recall over Epochs", "Epoch", ["Box_P", "Box_R"], ["#FF9CEE", "#FFF95B"], train_df, "4_precision_recall.png"),
    ("GPU Memory Usage over Epochs", "Epoch", "GPU_mem", "#64DFDF", train_df, "5_gpu_mem.png"),

    # From classwise_metrics.csv
    ("Per-Class Precision", "Class", "Precision", "#4CC9F0", class_df, "6_precision_per_class.png"),
    ("Per-Class Recall", "Class", "Recall", "#7209B7", class_df, "7_recall_per_class.png"),
    ("Per-Class mAP@0.5", "Class", "mAP50", "#3A0CA3", class_df, "8_map_per_class.png"),
    ("Per-Class mAP@0.5:0.95", "Class", "mAP50-95", "#58FCEC", class_df, "9_map_5095_per_class.png"),
]

# === Plot each graph ===
for title, x_col, y_col, color, df, filename in graph_definitions:
    plt.figure(figsize=(10, 6))

    if isinstance(y_col, list):  # Dual-line plots
        for y, c in zip(y_col, color):
            if y in df.columns:
                sns.lineplot(x=df[x_col], y=df[y], label=y, linewidth=2.5, color=c)
        plt.legend(loc="best", fontsize=10)

    else:  # Single line/bar
        if x_col == "Class":
            sns.barplot(x=df[x_col], y=df[y_col], color=color)
            plt.xticks(rotation=30, ha='right', fontsize=10)
        else:
            sns.lineplot(x=df[x_col], y=df[y_col], linewidth=2.5, color=color)

    # Titles and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_col, fontsize=13)
    plt.ylabel(y_col if isinstance(y_col, str) else "Metric", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(color='white')
    plt.yticks(color='white')
    sns.despine()

    # Save each graph
    output_path = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='black')
    plt.close()

print("✅ All YOLOv8 graphs saved to:", save_path)
