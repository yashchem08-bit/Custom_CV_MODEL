import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any

from model import CustomVisionModel
from dataset import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom Computer Vision Architecture")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to image dataset (with 'train' and 'val' subdirs)")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output target classes")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--img-size", type=int, default=64, help="Image resolution: 32 or 64 is fast & cool on CPU; 224 for GPU")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training set size (e.g. 2000) for fast, cool CPU training")
    parser.add_argument("--threads", type=int, default=None, help="Limit CPU threads (e.g. 2 or 4) to prevent overheating & fan blast")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cuda', 'cpu', or leave empty for auto-detect")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset mode for rapid pipeline testing")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision for high GPU throughput (safely bypassed on CPU)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100.0
    return {"loss": avg_loss, "accuracy": accuracy, "elapsed": elapsed}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100.0
    return {"val_loss": avg_loss, "val_accuracy": accuracy}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Device Configuration (Seamless CPU vs GPU handling)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"==========================================================")
    print(f"[Device Manager] Training on: {device.type.upper()}")
    if device.type == "cuda":
        print(f"   GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Mixed Precision (AMP): Enabled (FP16/BF16)")
    else:
        # Cap CPU threads to prevent thermal saturation and fan blasting
        cpu_threads = args.threads if args.threads else min(4, max(1, (os.cpu_count() or 4) // 2))
        torch.set_num_threads(cpu_threads)
        print(f"   CPU Threads: {torch.get_num_threads()} (capped to keep PC cool & quiet)")
        print(f"   Mixed Precision (AMP): Disabled for CPU stability")
    print(f"==========================================================")

    # 2. Data Loaders
    num_workers = 0 if os.name == 'nt' and device.type == 'cpu' else 2
    max_val_samples = int(args.max_samples * 0.25) if args.max_samples else None
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=args.img_size,
        num_classes=args.num_classes,
        use_synthetic=args.synthetic or (args.data_dir is None),
        max_train_samples=args.max_samples,
        max_val_samples=max_val_samples
    )

    # 3. Model Initialization
    model = CustomVisionModel(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized CustomVisionModel with {total_params / 1e6:.2f}M trainable parameters.")

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.2f}% | "
            f"Time: {train_metrics['elapsed']:.2f}s"
        )

        # Save Best Checkpoint
        if val_metrics["val_accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "num_classes": args.num_classes
            }, checkpoint_path)
            print(f" -> Best model saved with Val Acc: {best_val_acc:.2f}% to {checkpoint_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
