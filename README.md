# DeepFake-Image-Detection
Deepfake face image detection using EfficientNet-B0 transfer learning — 99.57% accuracy on 140k dataset

🔍 Deepfake Image Detection

A deep learning model to detect AI-generated (deepfake) face images 
using EfficientNet-B0 transfer learning trained on 140,000 real and 
fake face images.

🏆 Results
- Accuracy  : 99.57%
- AUC-ROC   : 0.9999
- F1 Score  : 99.57%

🛠️ Tech Stack
- PyTorch & EfficientNet-B0 (timm)
- Albumentations for augmentation
- Mixed Precision Training (AMP)
- Trained on Kaggle T4 GPU

📦 Dataset
140k Real and Fake Faces (Kaggle)
- 70,000 real faces (FFHQ)
- 70,000 fake faces (StyleGAN)

⚙️ Features
- Transfer learning with frozen backbone warmup
- Early stopping & automatic checkpointing
- Full evaluation metrics & plots
- Local prediction script
