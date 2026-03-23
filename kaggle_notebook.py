# ============================================================
# DEEPFAKE DETECTION — KAGGLE NOTEBOOK
# EfficientNet-B0 Transfer Learning
# Dataset: 140k Real and Fake Faces
# ============================================================

# ── CELL 1 — Install Dependencies ──
# !pip install -q timm albumentations

# ── CELL 2 — Imports ────────────────────────────────────────
import os, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.notebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# ── CELL 3 — Dataset ────────────────────────────────────────
BASE_PATH = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake"
CSV_PATH  = "/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces"
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]

def get_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(p=0.2),
        A.ImageCompression(quality_range=(70, 100), p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = 0 if row["label"] == 1 else 1   # 0=real, 1=fake
        img   = np.array(Image.open(f"{self.base_path}/{row['path']}").convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

train_ds = DeepfakeDataset(f"{CSV_PATH}/train.csv", BASE_PATH, get_train_transforms())
val_ds   = DeepfakeDataset(f"{CSV_PATH}/valid.csv", BASE_PATH, get_val_transforms())
test_ds  = DeepfakeDataset(f"{CSV_PATH}/test.csv",  BASE_PATH, get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")


# ── CELL 4 — Model ──────────────────────────────────────────
def build_model():
    model       = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(256, 1)
    )
    return model

model = build_model().to(DEVICE)
print(f"Model ready — Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ── CELL 5 — Training ───────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
scaler    = GradScaler()

os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

best_val_acc, patience_ctr, PATIENCE = 0.0, 0, 7

for epoch in range(1, 41):
    # ── Train ──
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            out  = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        preds      = (torch.sigmoid(out) > 0.5).float()
        t_correct += (preds == labels).sum().item()
        t_total   += labels.size(0)
        t_loss    += loss.item() * labels.size(0)

    # ── Validate ──
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} Val  "):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            with autocast():
                out  = model(images)
                loss = criterion(out, labels)
            preds      = (torch.sigmoid(out) > 0.5).float()
            v_correct += (preds == labels).sum().item()
            v_total   += labels.size(0)
            v_loss    += loss.item() * labels.size(0)

    train_acc = t_correct / t_total
    val_acc   = v_correct / v_total
    scheduler.step()

    print(f"Epoch {epoch:>2} | Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}%")

    # ── Save best ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ctr = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
            "arch":        "efficientnet_b0",
        }, "/kaggle/working/checkpoints/best_model.pth")
        print(f"  ✅ Best saved! val_acc={val_acc*100:.2f}%")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"⏹ Early stopping at epoch {epoch}")
            break

    # ── Periodic checkpoint every 5 epochs ──
    if epoch % 5 == 0:
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
            "arch":        "efficientnet_b0",
        }, f"/kaggle/working/checkpoints/checkpoint_epoch{epoch}.pth")

print(f"\n🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")


# ── CELL 6 — Evaluate on Test Set ───────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

ckpt = torch.load("/kaggle/working/checkpoints/best_model.pth")
model.load_state_dict(ckpt["model_state"])
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        with autocast():
            probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs >= 0.5).astype(int)

print(f"Accuracy : {accuracy_score(all_labels, preds)*100:.2f}%")
print(f"F1 Score : {f1_score(all_labels, preds)*100:.2f}%")
print(f"AUC-ROC  : {roc_auc_score(all_labels, all_probs):.4f}")
print("\n", classification_report(all_labels, preds, target_names=["Real", "Fake"]))


# ── CELL 7 — Save Model ─────────────────────────────────────
import shutil
shutil.copy("/kaggle/working/checkpoints/best_model.pth", "/kaggle/working/best_model.pth")
print(f"✅ Model saved!")
print(f"   Size: {os.path.getsize('/kaggle/working/best_model.pth') / 1024 / 1024:.1f} MB")
