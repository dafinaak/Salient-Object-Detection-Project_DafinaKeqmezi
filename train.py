import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import get_loaders
from sod_model import SODNet, SODNetImproved


def iou_score(pred, target, eps=1e-6, threshold=0.5):
    """IoU on binarized predictions - used as a *metric* (no gradients)."""
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def bce_iou_loss(pred, target, eps=1e-6):
    """Combined loss: BCE + 0.5 * (1 - soft IoU). Soft IoU keeps gradients flowing."""
    bce = F.binary_cross_entropy(pred, target)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return bce + 0.5 * (1 - iou.mean())


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"   [SAVE] checkpoint -> {os.path.basename(path)}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt.get("history", [])


def train(img_dir, mask_dir, ckpt_dir,
          improved=False,
          num_epochs=25, lr=1e-3, batch_size=16,
          early_stop_patience=5, img_size=128, seed=42):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(ckpt_dir, exist_ok=True)
    suffix = "_improved" if improved else ""
    last_ckpt = os.path.join(ckpt_dir, f"last{suffix}.pth")
    best_ckpt = os.path.join(ckpt_dir, f"best{suffix}.pth")
    log_csv = os.path.join(ckpt_dir, f"training_log{suffix}.csv")

    train_loader, val_loader, _ = get_loaders(
        img_dir, mask_dir, img_size=img_size, batch_size=batch_size, seed=seed,
    )

    ModelCls = SODNetImproved if improved else SODNet
    model = ModelCls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Model: {ModelCls.__name__} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    start_epoch, best_val_loss, history = 0, float("inf"), []
    if os.path.exists(last_ckpt):
        start_epoch, best_val_loss, history = load_checkpoint(
            last_ckpt, model, optimizer, device
        )
        print(f"[RESUME] Loaded checkpoint. Resuming from epoch {start_epoch} "
              f"(best_val_loss={best_val_loss:.4f})")
    else:
        print("[FRESH] No checkpoint found. Starting from epoch 0.")

    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        model.train()
        train_loss_sum, train_iou_sum, n = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = bce_iou_loss(preds, masks)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            train_loss_sum += loss.item() * bs
            train_iou_sum += iou_score(preds, masks) * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / n
        train_iou = train_iou_sum / n


        model.eval()
        val_loss_sum, val_iou_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = bce_iou_loss(preds, masks)
                bs = imgs.size(0)
                val_loss_sum += loss.item() * bs
                val_iou_sum += iou_score(preds, masks) * bs
                n += bs
        val_loss = val_loss_sum / n
        val_iou = val_iou_sum / n
        elapsed = time.time() - t0

        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} train_IoU={train_iou:.4f} "
              f"| val_loss={val_loss:.4f} val_IoU={val_iou:.4f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou,
        })

        save_checkpoint({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, last_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, best_ckpt)
            print(f"   [BEST] val_loss improved -> saved {os.path.basename(best_ckpt)}")
        else:
            epochs_no_improve += 1
            print(f"   no improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print(f"[EARLY STOP] val_loss has not improved for {early_stop_patience} epochs.")
                break

    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_iou", "val_loss", "val_iou"])
        w.writeheader()
        w.writerows(history)

    print(f"\nTraining done. Best val_loss = {best_val_loss:.4f}")
    print(f"Log saved to {log_csv}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Folder with .jpg images")
    parser.add_argument("--mask_dir", required=True, help="Folder with .png masks")
    parser.add_argument("--ckpt_dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--improved", action="store_true",
                        help="Train SODNetImproved (BatchNorm + skip connections) instead of baseline")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        ckpt_dir=args.ckpt_dir,
        improved=args.improved,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.patience,
    )
