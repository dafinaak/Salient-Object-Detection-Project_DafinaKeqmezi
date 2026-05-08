import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import get_loaders
from sod_model import SODNet, SODNetImproved


def compute_metrics(model, loader, device, threshold=0.5, eps=1e-6):
    """
    Compute mean IoU, Precision, Recall, F1, MAE across all images in ``loader``.
    Returns a dict.
    """
    iou_list, prec_list, rec_list, f1_list, mae_list = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds_bin = (preds > threshold).float()

            for i in range(imgs.size(0)):
                p = preds_bin[i].flatten()
                t = masks[i].flatten()

                tp = (p * t).sum()
                fp = (p * (1 - t)).sum()
                fn = ((1 - p) * t).sum()

                iou = (tp + eps) / (tp + fp + fn + eps)
                prec = (tp + eps) / (tp + fp + eps)
                rec = (tp + eps) / (tp + fn + eps)
                f1 = (2 * prec * rec + eps) / (prec + rec + eps)
                mae = (preds[i] - masks[i]).abs().mean()

                iou_list.append(iou.item())
                prec_list.append(prec.item())
                rec_list.append(rec.item())
                f1_list.append(f1.item())
                mae_list.append(mae.item())

    return {
        "IoU":       float(np.mean(iou_list)),
        "Precision": float(np.mean(prec_list)),
        "Recall":    float(np.mean(rec_list)),
        "F1":        float(np.mean(f1_list)),
        "MAE":       float(np.mean(mae_list)),
    }


def overlay_mask(img_np, mask_np, color=(255, 0, 0), alpha=0.5):
    """Blend ``mask_np`` (binary) onto ``img_np`` in the given color."""
    overlay = img_np.copy()
    overlay[mask_np > 0.5] = (
        (1 - alpha) * overlay[mask_np > 0.5] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def visualize_predictions(model, loader, device, n=6, save_path=None):
    """
    Plot N rows: [input | ground truth | prediction | overlay].
    """
    imgs, masks = next(iter(loader))
    imgs_dev = imgs.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(imgs_dev).cpu()

    n = min(n, imgs.size(0))
    fig, ax = plt.subplots(n, 4, figsize=(14, 3 * n))
    if n == 1:
        ax = ax[None, :]

    for i in range(n):
        img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_np = masks[i, 0].numpy()
        pred_np = (preds[i, 0].numpy() > 0.5).astype(np.float32)
        over_np = overlay_mask(img_np, pred_np)

        ax[i, 0].imshow(img_np);                ax[i, 0].set_title("Input")
        ax[i, 1].imshow(gt_np, cmap="gray");    ax[i, 1].set_title("Ground Truth")
        ax[i, 2].imshow(pred_np, cmap="gray");  ax[i, 2].set_title("Prediction")
        ax[i, 3].imshow(over_np);               ax[i, 3].set_title("Overlay")
        for a in ax[i]:
            a.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to a .pth checkpoint")
    parser.add_argument("--improved", action="store_true",
                        help="Use SODNetImproved architecture (must match the checkpoint)")
    parser.add_argument("--save_viz", default="sample_predictions.png",
                        help="Path to save the prediction visualization")
    parser.add_argument("--n_viz", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_loaders(args.img_dir, args.mask_dir)

    ModelCls = SODNetImproved if args.improved else SODNet
    model = ModelCls().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {ModelCls.__name__} from epoch {ckpt['epoch']}")

    metrics = compute_metrics(model, test_loader, device)
    print("\n--- Test set results ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    visualize_predictions(model, test_loader, device, n=args.n_viz, save_path=args.save_viz)


if __name__ == "__main__":
    main()
