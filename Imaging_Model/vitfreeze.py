# Swin Transformer Model for predicting LOS from Stony Brook COVID-19 CR Images
# Includes stratified cross-validation, preprocessing, training, and per-fold performance logging

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import nibabel as nib
import cv2
from torchvision import transforms
from skimage.transform import resize
import matplotlib.pyplot as plt
import logging
import torchxrayvision as xrv
import timm



# Setup logging
log_file = 'training_log.txt'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)

# Load metadata
df = pd.read_csv('/WAVE/projects/sd-2024-25-ai-pulmonary/CR/patient_dict.csv')
df = df[['subjectID', 'CR_image_paths', 'length_of_stay']].dropna()

# Create bins for stratification
los_freq = df['length_of_stay'].value_counts().sort_index()

total_patients = len(df)
target_bin_count = 8 if total_patients >= 800 else max(3, total_patients // 100)
min_bin_size = total_patients // target_bin_count

bin_edges = []
bin_total = 0
for los_day, freq in los_freq.items():
    bin_total += freq
    if bin_total >= min_bin_size:
        bin_edges.append(los_day)
        bin_total = 0

max_los = df['length_of_stay'].max()
if len(bin_edges) == 0 or bin_edges[-1] < max_los:
    bin_edges.append(max_los + 1)
if bin_edges[0] > 1:
    bin_edges = [0] + bin_edges

bin_edges = sorted(set(bin_edges))

labels = list(range(len(bin_edges) - 1))
df['stratify_bin'] = pd.cut(df['length_of_stay'], bins=bin_edges, labels=labels, right=False)

subjects = df['subjectID'].values
bins = df['stratify_bin'].astype(int).values  

#Preprocess
def preprocess_image(path):
    try:
        nii = nib.load(path)
        img = nii.get_fdata()
        img[img <= 500] = 0

        nonzero = img[img > 0]
        if nonzero.size == 0:
            logging.info(f"All pixels in {path} are below or equal to 500.")
            return None
        min_val, max_val = nonzero.min(), nonzero.max()
        img = (img - min_val) / (max_val - min_val + 1e-8)
        img[img == (0 - min_val) / (max_val - min_val + 1e-8)] = 0

        img = resize(img, (224, 224), preserve_range=True)
        img_uint8 = np.uint8(img * 255)

        if np.max(img_uint8) == 0:
            logging.info(f"CLAHE skipped for {path}, image is all zeros after masking.")
            return None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)

        img_tensor = torch.tensor(img_clahe, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1) 
        return img_tensor
    
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# Dataset
class CRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.stack(images)  
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        if img.ndimension() == 2:
            img = img.unsqueeze(0)  

        if self.transform:
            img = self.transform(img)

        return img, label





# Transforms
use_augmentation = True 
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Normalize(mean=[0.5], std=[0.5])
]) if use_augmentation else transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class VitRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.3):
        super(VitRegressor, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)  # No classifier head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.regressor(features)
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 25
batch_size = 32
lr = 1e-4
k = 5
run_single_fold = False

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
rmse_scores, c_index_scores = [], []
os.makedirs('vit_checkpoints', exist_ok=True)

all_train_losses = []
all_val_losses = []
c_index_per_epoch_all_folds = []

fold_range = range(1) if run_single_fold else range(k)

for fold in fold_range:
    print(f"Fold {fold+1}/{k} started")

    train_idx, val_idx = list(skf.split(subjects, bins))[fold]
    train_subjects = subjects[train_idx]
    val_subjects = subjects[val_idx]

    train_df = df[df['subjectID'].isin(train_subjects)]
    val_df = df[df['subjectID'].isin(val_subjects)]

    train_imgs, train_labels = [], []
    for _, row in train_df.iterrows():
        img = preprocess_image(row['CR_image_paths'])
        if img is not None:
            train_imgs.append(img.squeeze(0))  
            train_labels.append(row['length_of_stay'])

    val_imgs, val_labels = [], []
    for _, row in val_df.iterrows():
        img = preprocess_image(row['CR_image_paths'])
        if img is not None:
            val_imgs.append(img.squeeze(0)) 
            val_labels.append(row['length_of_stay'])


    train_dataset = CRDataset(train_imgs, train_labels, transform=train_transforms)
    val_dataset = CRDataset(val_imgs, val_labels, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)



    class RMSELoss(nn.Module):
        def __init__(self):
            super(RMSELoss, self).__init__()
            self.mse = nn.MSELoss()

        def forward(self, yhat, y):
            return torch.sqrt(self.mse(yhat, y))
        

    criterion = RMSELoss()
    model = VitRegressor().to(device)

    
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)
    scheduler = None  
    
    train_losses, val_losses = [], []
    c_index_per_epoch = []

    unfreeze_epoch = 5  

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            print(f"Unfreezing backbone at epoch {epoch+1}")
            logging.info(f"Unfreezing backbone at epoch {epoch+1}")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=0.05)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - unfreeze_epoch)
            
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        train_rmse = total_loss / len(train_loader.dataset)
        train_losses.append(train_rmse)

        # Validation
        model.eval()
        total_val_loss, val_preds, val_targets = 0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * labels.size(0)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(labels.squeeze().cpu().numpy())

        val_rmse = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_rmse)

        c_index_epoch = concordance_index(val_targets, val_preds)
        c_index_per_epoch.append(c_index_epoch)

        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1} - LR: {current_lr:.6f}")


        if epoch == num_epochs - 1:
            pred_df = pd.DataFrame({
                'subjectID': val_subjects,
                'true_los': val_targets,
                'predicted_los': val_preds
            })
            pred_df.to_csv(f"vit_checkpoints/predictions_fold_{fold+1}.csv", index=False)

            model_path = f"vit_checkpoints/model_fold_{fold+1}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)

            log_msg = (f"Epoch {epoch+1} - Train RMSE: {train_rmse:.4f} | "
                       f"Val RMSE: {val_rmse:.4f} | C-index: {c_index_epoch:.4f}")
            logging.info(log_msg)
            print(log_msg)


    # Save per-fold metrics
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    c_index_per_epoch_all_folds.append(c_index_per_epoch)
    rmse_scores.append(val_losses[-1])
    c_index_scores.append(c_index_per_epoch[-1])
    logging.info(f"Fold {fold+1} C-index: {c_index_per_epoch[-1]:.4f}")
    print(f"Fold {fold+1} C-index: {c_index_per_epoch[-1]:.4f}")

    # Save per-epoch RMSE
    rmse_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs+1)),
        'train_rmse': train_losses,
        'val_rmse': val_losses
    })
    rmse_df.to_csv(f"vit_checkpoints/rmse_fold_{fold+1}.csv", index=False)

    # Save per-epoch C-index
    c_index_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs+1)),
        'c_index': c_index_per_epoch
    })
    c_index_df.to_csv(f"vit_checkpoints/c_index_fold_{fold+1}.csv", index=False)

    # Loss plots
    plt.figure()
    plt.plot(train_losses, label="Train RMSE")
    plt.plot(val_losses, label="Val RMSE")
    plt.legend()
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.savefig(f"vit_checkpoints/loss_plot_fold_{fold+1}.png")
    plt.close()

# Average loss curves
avg_train_loss = np.mean(np.array(all_train_losses), axis=0)
avg_val_loss = np.mean(np.array(all_val_losses), axis=0)

plt.figure()
plt.plot(avg_train_loss, label='Avg Train RMSE')
plt.plot(avg_val_loss, label='Avg Val RMSE')
plt.title("Average Loss Curve Across Folds")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("vit_checkpoints/average_loss_curve.png")
plt.close()

# Log summary
logging.info(f"Average RMSE Scores: {rmse_scores}")
logging.info(f"Average C-index Scores: {c_index_scores}")
print("Average RMSE Scores:", rmse_scores)
print("Average C-index Scores:", c_index_scores)

# Combine prediction
all_preds = []
for fold in fold_range:
    pred_path = f"vit_checkpoints/predictions_fold_{fold+1}.csv"
    if os.path.exists(pred_path):
        fold_preds = pd.read_csv(pred_path)
        fold_preds['fold'] = fold + 1
        all_preds.append(fold_preds)

if all_preds:
    combined_df = pd.concat(all_preds, ignore_index=True)
    combined_df.to_csv("vit_checkpoints/all_fold_predictions.csv", index=False)
    print("Saved combined predictions to vit_checkpoints/all_fold_predictions.csv")
