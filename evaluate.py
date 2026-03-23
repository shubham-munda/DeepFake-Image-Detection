# ============================================================
# evaluate.py — Full Evaluation on Kaggle Test Set
# Outputs: Accuracy, F1, AUC-ROC, Confusion Matrix, ROC Curve
# Run this on Kaggle after training
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)


# ── Config ───────────────────────────────────────────────────
BASE_PATH  = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake"
CSV_PATH   = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces"
CKPT_PATH  = "/kaggle/working/checkpoints/best_model.pth"
OUT_DIR    = "/kaggle/working/results"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

os.makedirs(OUT_DIR, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = 0 if row["label"] == 1 else 1
        img   = np.array(Image.open(f"{self.base_path}/{row['path']}").convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

test_ds     = DeepfakeDataset(f"{CSV_PATH}/test.csv", BASE_PATH, transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
print(f"Test set: {len(test_ds):,} images")


# ── Model ────────────────────────────────────────────────────
def build_model():
    model       = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(256, 1)
    )
    return model

ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model = build_model()
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE).eval()
print(f"✅ Model loaded (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']*100:.2f}%)")


# ── Inference ────────────────────────────────────────────────
all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE, non_blocking=True)
        with autocast():
            probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs >= 0.5).astype(int)


# ── Metrics ──────────────────────────────────────────────────
acc  = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds, zero_division=0)
rec  = recall_score(all_labels, preds, zero_division=0)
f1   = f1_score(all_labels, preds, zero_division=0)
auc  = roc_auc_score(all_labels, all_probs)

print(f"\n{'='*50}")
print(f"  TEST SET RESULTS")
print(f"{'='*50}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"{'='*50}\n")
print(classification_report(all_labels, preds, target_names=["Real", "Fake"]))


# ── Plot 1: Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(all_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.show()
print(f"✅ Confusion matrix saved")


# ── Plot 2: ROC Curve ────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#e63946", lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.show()
print(f"✅ ROC curve saved")


# ── Plot 3: Confidence Distribution ──────────────────────────
real_probs = all_probs[all_labels == 0]
fake_probs = all_probs[all_labels == 1]
plt.figure(figsize=(7, 4))
plt.hist(real_probs, bins=50, alpha=0.6, color="#2a9d8f", label="Real")
plt.hist(fake_probs, bins=50, alpha=0.6, color="#e63946", label="Fake")
plt.axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
plt.xlabel("Predicted Probability (Fake)")
plt.ylabel("Count")
plt.title("Confidence Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confidence_dist.png", dpi=150)
plt.show()
print(f"✅ Confidence distribution saved")

print(f"\n📁 All plots saved to {OUT_DIR}")
