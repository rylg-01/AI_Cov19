import os
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision import transforms
from skimage.transform import resize
import timm
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Image preprocessing
def preprocess_xray(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = resize(img, (224, 224), preserve_range=True)
        img_uint8 = np.uint8(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        img_tensor = torch.tensor(img_clahe, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)  # (3, H, W)
        return img_tensor
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# Directories
covid_dir = "/WAVE/projects/sd-2024-25-ai-pulmonary/pos_neg_database/COVID-19_Radiography_Dataset/COVID/images"
normal_dir = "/WAVE/projects/sd-2024-25-ai-pulmonary/pos_neg_database/COVID-19_Radiography_Dataset/Normal/images"
model_dir = "vit_checkpoints"
plot_dir = "/WAVE/projects/sd-2024-25-ai-pulmonary/CR/runs/pos_neg_analysis/pos_neg_plots"
os.makedirs(plot_dir, exist_ok=True)

# Load img
def load_group_images(image_dir):
    tensors = []
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        img = preprocess_xray(fpath)
        if img is not None:
            tensors.append(img)
    if not tensors:
        return None
    return torch.stack(tensors)

covid_images = load_group_images(covid_dir)
normal_images = load_group_images(normal_dir)
print(f"COVID images: {covid_images.shape[0]}, Normal images: {normal_images.shape[0]}")

# Run model
def run_in_batches(model, images, device, batch_size=32):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            out = model(batch).squeeze().cpu().numpy()
            preds.extend(out if isinstance(out, np.ndarray) else [out])
    return np.array(preds)

#Evaluate each fold
results = []
for fold in range(1, 6):  
    model_path = os.path.join(model_dir, f"model_fold_{fold}_epoch_25.pth")
    if not os.path.exists(model_path):
        print(f"Model for fold {fold} not found, skipping.")
        continue

    model = VitRegressor(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    covid_preds = run_in_batches(model, covid_images, device, batch_size=16)
    normal_preds = run_in_batches(model, normal_images, device, batch_size=16)

    # Labels + predictions
    all_preds = np.concatenate([covid_preds, normal_preds])
    labels = np.concatenate([np.ones(len(covid_preds)), np.zeros(len(normal_preds))])  # 1=COVID, 0=Normal

    # best threshold
    best_thresh, best_acc = None, 0
    for t in np.arange(0, np.max(all_preds), 0.1):
        pred_labels = (all_preds > t).astype(int)
        acc = accuracy_score(labels, pred_labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    
    final_preds = (all_preds > best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(labels, final_preds)
    prec = precision_score(labels, final_preds, zero_division=0)
    rec = recall_score(labels, final_preds, zero_division=0)
    f1 = f1_score(labels, final_preds, zero_division=0)
    cm = confusion_matrix(labels, final_preds)
    auc_value = roc_auc_score(labels, all_preds)

    results.append({
        "fold": fold,
        "best_threshold_LOS": round(best_thresh, 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "AUC": round(auc_value, 4),
        "mean_LOS_covid": round(np.mean(covid_preds), 2),
        "mean_LOS_normal": round(np.mean(normal_preds), 2),
        "covid_preds": covid_preds,
        "normal_preds": normal_preds
    })

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["accuracy"].idxmax()]
print("\n=== Results per fold ===")
print(results_df)
print("\n=== Best Fold ===")
print(best_row)

# LOS Distribution Plots
covid_los = best_row['covid_preds']
normal_los = best_row['normal_preds']

# Combine into DataFrame for plotting
plot_df = pd.DataFrame({
    'LOS': np.concatenate([covid_los, normal_los]),
    'Group': ['COVID']*len(covid_los) + ['Normal']*len(normal_los)
})

# Histogram
plt.figure(figsize=(10,6))
sns.histplot(plot_df[plot_df['Group']=='COVID']['LOS'], color='red', kde=True, stat="density", label='COVID', bins=50)
sns.histplot(plot_df[plot_df['Group']=='Normal']['LOS'], color='blue', kde=True, stat="density", label='Normal', bins=50)
plt.axvline(best_row['best_threshold_LOS'], color='black', linestyle='--', label='Threshold')
plt.xlabel("Length of Stay (LOS)")
plt.ylabel("Density")
plt.title("LOS Distribution: COVID vs Normal")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "LOS_histogram.png"))
plt.close()

# Visualization
plt.figure(figsize=(8,6))
sns.violinplot(x='Group', y='LOS', data=plot_df, inner='quartile', palette={'COVID':'red','Normal':'blue'})
sns.stripplot(x='Group', y='LOS', data=plot_df, color='black', alpha=0.3)
plt.title("LOS Violin / Boxplot: COVID vs Normal")
plt.ylabel("Length of Stay (LOS)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "LOS_violin_boxplot.png"))
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(
    np.concatenate([np.ones(len(best_row['covid_preds'])), np.zeros(len(best_row['normal_preds']))]),
    np.concatenate([best_row['covid_preds'], best_row['normal_preds']])
)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Best Fold {int(best_row['fold'])})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "ROC_curve.png"))
plt.close()
print(f"Plots saved to {plot_dir}")
