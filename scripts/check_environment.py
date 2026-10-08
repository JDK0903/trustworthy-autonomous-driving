"""Verify imports and an actual ResNet18 CUDA forward/backward pass."""
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib
import numpy
import pandas
import PIL
import pytest
import sklearn
import torch
import torchvision
import yaml


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable in this Python environment")
    torch.manual_seed(42)
    model = torchvision.models.resnet18(weights=None).cuda()
    model.fc = torch.nn.Linear(model.fc.in_features, 5).cuda()
    images = torch.randn(2, 3, 224, 448, device="cuda")
    targets = torch.zeros(2, 5, device="cuda")
    logits = model(images)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    torch.cuda.synchronize()
    assert logits.shape == (2, 5)
    assert torch.isfinite(loss).item()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all().item()
    report = {"python":platform.python_version(), "torch":torch.__version__,
              "torchvision":torchvision.__version__, "cuda_runtime":torch.version.cuda,
              "gpu":torch.cuda.get_device_name(0), "gpu_capability":torch.cuda.get_device_capability(0),
              "output_shape":list(logits.shape), "loss":loss.item(),
              "forward_backward_pass":"passed", "weights":"random; no pretrained download or training"}
    path = Path(__file__).resolve().parents[1] / "outputs/environment_verification.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
