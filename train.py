"""
train.py
--------
Full training script for the Salient Object Detection model.

Features:
- Forward + backward + optimizer step
- Validation after each epoch
- Saves the best model (lowest validation loss)
- Saves a "last" checkpoint each epoch with optimizer state for resuming
  (Bonus task: automatically resumes from checkpoints/last.pt if present)
- tqdm progress bars
- Logs per-epoch loss and validation metrics

Usage:
    python train.py
"""

import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model import SODModel
from utils import (
    BCE_IoU_Loss,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
)


# ---------------------------- Single-epoch helpers ---------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Run one full pass over the training data."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)              # forward pass
        loss = criterion(preds, masks)     # BCE + 0.5 * (1 - IoU)
        loss.backward()                    # backward pass
        optimizer.step()                   # update weights

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    """Evaluate on the validation set and return loss + metrics."""
    model.eval()
    running_loss = 0.0
    metric_sums = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)
        running_loss += loss.item() * images.size(0)

        m = compute_metrics(preds, masks)
        for k in metric_sums:
            metric_sums[k] += m[k]
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{m['iou']:.3f}")

    avg_loss = running_loss / len(loader.dataset)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


# --------------------------------- Main --------------------------------- #
def main(args):
    # Reproducibility
    torch.manual_seed(args.seed)

    # Device selection (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Model, loss, optimizer
    model = SODModel().to(device)
    criterion = BCE_IoU_Loss(iou_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Bookkeeping
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    last_path = os.path.join(args.checkpoint_dir, "last.pt")

    # ----- Bonus: auto-resume from last checkpoint if it exists ----- #
    start_epoch, best_val_loss = load_checkpoint(last_path, model, optimizer, device)
    if start_epoch > 0:
        print(f"==> Resuming training from epoch {start_epoch + 1}")
    start_epoch = start_epoch  # already 0 if no checkpoint

    # Early-stopping bookkeeping
    epochs_without_improvement = 0

    # ----- Main training loop ----- #
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | "
            f"R: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        # ----- Save "last" checkpoint every epoch (for resume) ----- #
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, last_path)
        print(f"  [checkpoint] Saved last checkpoint -> {last_path}")

        # ----- Save best model when validation loss improves ----- #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_path)
            print(f"  [checkpoint] New best! Saved best model -> {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        # ----- Simple early stopping ----- #
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


# --------------------------------- CLI --------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--masks_dir", type=str, default="dataset/masks")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7,
                        help="Stop if val loss doesn't improve for N epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)