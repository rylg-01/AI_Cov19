# Independent model evaluation on COVID patients split by severity (level 1, level 2)
# Saved model performance metrics

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.utils import resample

import cv2
from skimage.transform import resize
import timm
import matplotlib.pyplot as plt
import torch.nn.functional as F


csv_path = "/WAVE/projects/sd-2024-25-ai-pulmonary/ricord_images/a_ricord_image_severity.csv"
fold_model_dir = "vit_checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fixed_threshold = 7  

# Preprocessing (png)
def preprocess_png_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {path}")
            return None

        img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)

        img_tensor = torch.tensor(img_clahe, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)  # (3, 224, 224)
        return img_tensor
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


class VitRegressor(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.3):
        super(VitRegressor, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.regressor(features)
        return out

# Load Dataset
df = pd.read_csv(csv_path)
severity_map = {'level1': 1, 'level2': 2}
df['severity_num'] = df['severity'].map(severity_map)

print(f"📥 Loaded {len(df)} images for evaluation.")
num_before = len(df)
df = df.dropna(subset=['severity_num'])
num_after = len(df)
print(f"Dropped {num_before - num_after} rows with invalid or missing severity.")

valid_data = df[['image_path', 'severity_num']].copy()

level1_df = valid_data[valid_data['severity_num'] == 1]
level2_df = valid_data[valid_data['severity_num'] == 2]
print(f"Level 1 samples: {len(level1_df)}, Level 2 samples: {len(level2_df)}")


# Preprocess images
image_tensors = []
ground_truth = []
image_paths = []

for _, row in df.iterrows():
    tensor_img = preprocess_png_image(row["image_path"])
    if tensor_img is not None:
        image_tensors.append(tensor_img)
        ground_truth.append(row["severity_num"])
        image_paths.append(row["image_path"])


if len(image_tensors) == 0:
    raise RuntimeError("No valid images were loaded. Check image paths and preprocessing.")

images_tensor = torch.stack(image_tensors).to(device)
ground_truth = np.array(ground_truth)


results = []
best_avg_f1 = 0.0
best_model_path = None

for fold in range(1, 6):
    model_files = [f for f in os.listdir(fold_model_dir) if f"model_fold_{fold}_" in f and f.endswith(".pth")]
    if not model_files:
        print(f"No model found for fold {fold}")
        continue

    model_path = os.path.join(fold_model_dir, sorted(model_files)[-1])
    model = VitRegressor(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds_los = model(images_tensor).squeeze().cpu().numpy()

    preds_severity = np.where(preds_los < fixed_threshold, 1, 2)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(ground_truth, preds_severity, labels=[1,2], zero_division=0)
    avg_f1 = np.mean(f1)

    # Binary F1 for level2
    binary_f1_level2 = f1_score(ground_truth, preds_severity, average="binary", pos_label=2, zero_division=0)

    # ROC AUC (binary)
    try:
        roc_auc = roc_auc_score((ground_truth==2).astype(int), preds_los)
    except Exception:
        roc_auc = np.nan

    results.append({
        "fold": fold,
        "model_path": model_path,
        "precision_class1": precision[0],
        "recall_class1": recall[0],
        "f1_class1": f1[0],
        "support_class1": support[0],
        "precision_class2": precision[1],
        "recall_class2": recall[1],
        "f1_class2": f1[1],
        "support_class2": support[1],
        "avg_f1_macro": avg_f1,
        "binary_f1_level2": binary_f1_level2,
        "roc_auc_level2": roc_auc,
        "threshold": fixed_threshold
    })

    print(f"Fold {fold}: Avg F1={avg_f1:.4f}, Binary F1 (level2)={binary_f1_level2:.4f}, ROC AUC={roc_auc:.4f}")

    if avg_f1 > best_avg_f1:
        best_avg_f1 = avg_f1
        best_model_path = model_path


model = VitRegressor(pretrained=False).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

with torch.no_grad():
    predicted_days = model(images_tensor).squeeze().cpu().numpy()

thresholds = list(range(1, 15))
best_threshold = None
best_macro_f1 = -1
best_preds = None
accuracies = []

for threshold in thresholds:
    preds_severity = (predicted_days >= threshold).astype(int)+1
    precision, recall, f1, support = precision_recall_fscore_support(ground_truth, preds_severity, labels=[1,2], zero_division=0)
    avg_f1 = np.mean(f1)
    acc = np.mean(preds_severity==ground_truth)
    accuracies.append(acc)

    # Binary F1 level2
    binary_f1_level2 = f1_score(ground_truth, preds_severity, average="binary", pos_label=2, zero_division=0)

    print(f"Threshold {threshold}: Avg F1={avg_f1:.4f}, Binary F1 (level2)={binary_f1_level2:.4f}, Accuracy={acc:.4f}")

    if avg_f1 > best_macro_f1:
        best_macro_f1 = avg_f1
        best_threshold = threshold
        best_preds = preds_severity.copy()

print(f"\nBest Threshold by Avg Macro F1: {best_threshold}, Avg F1={best_macro_f1:.4f}")


predicted_severity = best_preds
true_severity = ground_truth

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    true_severity, predicted_severity, labels=[1,2], zero_division=0
)

# Binary F1 for level2
binary_f1_level2 = f1_score(
    true_severity, predicted_severity, average="binary", pos_label=2, zero_division=0
)

# AUC
roc_auc = roc_auc_score((true_severity==2).astype(int), predicted_days)
bal_acc = balanced_accuracy_score(true_severity, predicted_severity)

# Confusion matrix 
cm = confusion_matrix(true_severity, predicted_severity, labels=[1,2])
tn, fp, fn, tp = cm.ravel()

# Binary metrics (level2 = positive class)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\Classification Report")
print(f"Class 1 (level1): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}, Support={support[0]}")
print(f"Class 2 (level2): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}, Support={support[1]}")

print("\n=== Binary (level2 as Positive) ===")
print(f"Binary F1: {binary_f1_level2:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision (PPV): {precision_pos:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=["True level1", "True level2"], columns=["Pred level1", "Pred level2"]))


# save result
save_folder = "sev_eval_checkpoints"
os.makedirs(save_folder, exist_ok=True)

# Fold metrics
fold_results_df = pd.DataFrame(results)
fold_results_path = os.path.join(save_folder, "los_eval_fold_metrics.csv")
fold_results_df.to_csv(fold_results_path, index=False)
print(f"Saved fold metrics to {fold_results_path}")
threshold_results_df = pd.DataFrame({"threshold": thresholds, "accuracy": accuracies})
threshold_results_path = os.path.join(save_folder, "threshold_accuracies.csv")
threshold_results_df.to_csv(threshold_results_path, index=False)
print(f"Saved threshold accuracy results to {threshold_results_path}")

# Plot accuracy vs threshold
plt.figure(figsize=(8,5))
plt.plot(thresholds, accuracies, marker='o', linestyle='-')
plt.title("Accuracy vs Threshold")
plt.xlabel("Threshold (days)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(save_folder, "accuracy_vs_threshold.png")
plt.savefig(plot_path)
plt.show()
print(f"Accuracy plot saved to {plot_path}")

print(f"Best Model: {best_model_path}")
print(f"Best Threshold by Avg Macro F1: {best_threshold}, Avg F1 Score: {best_macro_f1:.4f}")


# Roc curve
fpr, tpr, thresholds_roc = roc_curve((true_severity == 2).astype(int), predicted_days)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Severity Level2 vs Level1")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(save_folder, "roc_curve.png")
plt.savefig(roc_path)
plt.show()
print(f"📈 ROC curve saved to {roc_path}")



predicted_severity_labels = np.where(predicted_days >= best_threshold, "level2", "level1")
actual_severity_labels = np.where(ground_truth == 2, "level2", "level1")

detailed_df = pd.DataFrame({
    "image_path": image_paths,
    "predicted_los": predicted_days,
    "predicted_severity": predicted_severity_labels,
    "actual_severity": actual_severity_labels
})

detailed_csv_path = os.path.join(save_folder, "best_threshold_predictions.csv")
detailed_df.to_csv(detailed_csv_path, index=False)
print(f"Saved detailed predictions to {detailed_csv_path}")
