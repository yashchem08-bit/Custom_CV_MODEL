import os
import argparse
import torch
from model import CustomVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export Trained Model to ONNX & TorchScript")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth checkpoint")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output target classes")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--output-dir", type=str, default="./exports", help="Directory to save exported formats")
    return parser.parse_args()


def export_model():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Instantiate Model
    model = CustomVisionModel(num_classes=args.num_classes)
    model.eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Export] Loaded weights from {args.checkpoint}")
    else:
        print("[Export] No checkpoint specified or found. Exporting base architecture with initial weights.")

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    # 2. Export TorchScript (JIT Trace)
    ts_path = os.path.join(args.output_dir, "model.torchscript.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(ts_path)
    print(f"[Export] Saved TorchScript model: {ts_path}")

    # 3. Export ONNX (for optimized CPU/Cloud serving with ONNX Runtime / Triton)
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_images"],
        output_names=["class_logits"],
        dynamic_axes={
            "input_images": {0: "batch_size"},
            "class_logits": {0: "batch_size"}
        }
    )
    print(f"[Export] Saved ONNX model: {onnx_path}")
    print("[Export] Export pipeline completed successfully.")


if __name__ == "__main__":
    export_model()
