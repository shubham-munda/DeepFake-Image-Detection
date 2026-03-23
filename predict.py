# ============================================================
# predict.py — Run Deepfake Detection on Your Laptop
# Usage:
#   Single image : python predict.py photo.jpg
#   Folder       : python predict.py test_images/
# ============================================================

import torch
import torch.nn as nn
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import sys, os


# ── Build Model ─────────────────────────────────────────────
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


# ── Load Checkpoint ─────────────────────────────────────────
def load_model(checkpoint_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device).eval()
    print(f"✅ Model loaded! Device: {device}")
    return model, device


# ── Predict Single Image ─────────────────────────────────────
def predict(model, device, image_path):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    img    = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    return {
        "file"      : os.path.basename(image_path),
        "prediction": "FAKE" if prob >= 0.5 else "REAL",
        "fake_prob" : round(prob * 100, 2),
        "real_prob" : round((1 - prob) * 100, 2),
        "confidence": round(max(prob, 1 - prob) * 100, 2),
    }


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "best_model.pth"
    IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_images"

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Make sure best_model.pth is in the same folder as predict.py")
        sys.exit(1)

    model, device = load_model(MODEL_PATH)

    # ── Single Image ──
    if os.path.isfile(IMAGE_PATH):
        r     = predict(model, device, IMAGE_PATH)
        emoji = "🔴" if r["prediction"] == "FAKE" else "🟢"
        print(f"\n{'='*45}")
        print(f"  Image     : {r['file']}")
        print(f"  Result    : {emoji} {r['prediction']}")
        print(f"  Fake prob : {r['fake_prob']}%")
        print(f"  Real prob : {r['real_prob']}%")
        print(f"  Confidence: {r['confidence']}%")
        print(f"{'='*45}")

    # ── Folder of Images ──
    elif os.path.isdir(IMAGE_PATH):
        images = [f for f in os.listdir(IMAGE_PATH)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            print(f"❌ No images found in {IMAGE_PATH}")
            sys.exit(1)

        print(f"\nProcessing {len(images)} images...\n")
        fake_count, real_count = 0, 0

        for img_name in sorted(images):
            r     = predict(model, device, os.path.join(IMAGE_PATH, img_name))
            emoji = "🔴" if r["prediction"] == "FAKE" else "🟢"
            print(f"{emoji} {r['prediction']} | {r['file']:<35} | "
                  f"Fake: {r['fake_prob']:>6}% | Conf: {r['confidence']}%")
            if r["prediction"] == "FAKE":
                fake_count += 1
            else:
                real_count += 1

        print(f"\n{'='*45}")
        print(f"  Total : {len(images)} images")
        print(f"  🔴 Fake : {fake_count}")
        print(f"  🟢 Real : {real_count}")
        print(f"{'='*45}")

    else:
        print(f"❌ Path not found: {IMAGE_PATH}")
