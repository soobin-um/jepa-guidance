#!/usr/bin/env python3
import os
os.environ["XFORMERS_DISABLED"] = "1"  # Disable xformers to avoid inplace op issues with jacobian

import glob
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub
import csv
from torch.autograd.functional import jacobian


def load_image(path: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Load image, resize to 224x224, and normalize with ImageNet stats.
    Returns tensor ready for DINOv2 backbone (not wrapper).
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W) in [0, 1]
    
    # Resize to 224x224 (DINOv2 native resolution)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    
    # ImageNet normalization
    x = (x - mean) / std
    
    return x


def jepa_score(backbone, images, eps=1e-6):
    J = jacobian(lambda x: backbone(x).sum(0), inputs=images)
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2)  # (B, D, N) where N = 3*224*224
        svdvals = torch.linalg.svdvals(J)
        score = svdvals.clip_(eps).log_().sum(1)  # (B,)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="results")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path (default: metrics/<basedir>.csv)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Disable memory-efficient attention backends that cause issues with jacobian
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Load JEPA backbone (without wrapper - we handle preprocessing separately)
    backbone = hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device).eval()
    
    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Image list (support both jpg and png)
    img_paths = sorted(glob.glob(os.path.join(args.img_root, "*.jpg")) +
                       glob.glob(os.path.join(args.img_root, "*.png")))
    
    if len(img_paths) == 0:
        raise RuntimeError("No images found")

    if args.output_csv:
        csv_path = args.output_csv
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        metrics_dir = os.path.join(project_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        basedir = os.path.basename(os.path.normpath(args.img_root))
        csv_path = os.path.join(metrics_dir, f"{basedir}_jepa.csv")

    # Compute JEPA scores & save to CSV
    all_scores = []
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "jepa_score"])
        
        for p in tqdm(img_paths, desc="JEPA scoring"):
            # Load images
            x = load_image(p, mean.cpu(), std.cpu()).to(device)
            
            # Compute JEPA scores
            score = jepa_score(backbone, x, eps=args.eps)
            score_val = score.item()
            all_scores.append(score_val)
            
            filename = os.path.basename(p)
            writer.writerow([filename, f"{score_val:.10f}"])
            del x

        # Write mean and std at the end
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        writer.writerow([])
        writer.writerow(["mean", f"{mean_score:.10f}"])
        writer.writerow(["std", f"{std_score:.10f}"])

    print(f"[Done] Saved {len(img_paths)} scores to {csv_path}")
    print(f"       Mean: {mean_score:.6f}, Std: {std_score:.6f}")


if __name__ == "__main__":
    main()