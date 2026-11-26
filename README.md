# Custom PyTorch Computer Vision Architecture & Workflow

A lightweight, modular, and production-ready PyTorch Computer Vision framework built from scratch. Designed to train and run seamlessly on both **CPU-only environments** and **High-Performance Cloud GPUs** (NVIDIA CUDA).

---

## 1. Architecture Highlights

* **Backbone:** Multi-stage Inverted Residual Network with Depthwise Separable Convolutions (inspired by MobileNetV3 / EfficientNet).
* **Attention:** Squeeze-and-Excitation (SE) channel-attention modules for dynamic feature recalibration.
* **Size:** ~2.8 Million parameters. Small enough to train quickly on a standard CPU, yet deep enough to achieve high accuracy.
* **Dual Execution Mode:**
  * **GPU / Cloud:** Automatically activates `torch.amp.autocast` (FP16/BF16) and pinned memory for maximum throughput.
  * **CPU:** Automatically drops AMP overhead and uses optimized single/multi-thread CPU kernels.

---

## 2. Directory Layout

```text
custom_cv_model/
├── requirements.txt         # Dependencies
├── README.md                # Documentation & Guide
└── src/
    ├── model.py             # Custom neural network architecture
    ├── dataset.py           # Augmentations & dataset loaders (with synthetic mode)
    ├── train.py             # Device-agnostic training loop with checkpointing
    └── export.py            # ONNX and TorchScript export pipeline
```

---

## 3. Quickstart & Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Instant Pipeline Verification (CPU or GPU)
You can test the entire training loop right away without downloading any external datasets using `--synthetic`:
```bash
python src/train.py --epochs 3 --batch-size 16 --synthetic
```

### Step 3: Training on Your Custom Dataset
Organize your image dataset into standard class directories:
```text
my_dataset/
├── train/
│   ├── class_cat/
│   └── class_dog/
└── val/
    ├── class_cat/
    └── class_dog/
```

Run training on **CPU (Cool & Fast Mode)**:
```bash
# Trains in ~1-2 minutes without overheating or fan blast
python src/train.py --data-dir "data/cifar10" --num-classes 10 --img-size 32 --max-samples 2000 --threads 2 --batch-size 32 --epochs 5
```

Run training on **Cloud GPU (Full Dataset)**:
```bash
python src/train.py --data-dir "data/cifar10" --num-classes 10 --img-size 64 --batch-size 64 --epochs 10 --device cuda
```

### Step 4: Export to Production (ONNX / TorchScript)
Convert your trained weights into ONNX for high-performance deployment:
```bash
python src/export.py --checkpoint "./checkpoints/best_model.pth" --num-classes 2
```
This generates:
* `exports/model.onnx` (Ready for ONNX Runtime / Triton / OpenVINO)
* `exports/model.torchscript.pt` (Ready for C++ / mobile / LibTorch)
