"""
utils.py
--------
Shared utility functions used across training, evaluation, and the demo.

Contents:
- IoU loss helper and combined BCE + 0.5 * (1 - IoU) loss
- Evaluation metrics: IoU, Precision, Recall, F1
- Visualization helper to plot input/GT/prediction/overlay
- Checkpoint save / load helpers (used by train.py to support resume)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================== LOSS =============================== #
def iou_score(pred, target, eps=1e-6):
    """
    Soft IoU (works on probabilities, differentiable).
    pred, target: tensors of shape (B, 1, H, W) with values in [0, 1].
    Returns a scalar IoU averaged over the batch.
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


class BCE_IoU_Loss(nn.Module):
    """
    Combined loss as required by the project:
        L = BCE(pred, target) + 0.5 * (1 - IoU(pred, target))
    """

    def __init__(self, iou_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou_weight = iou_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        iou_loss = 1.0 - iou_score(pred, target)
        return bce_loss + self.iou_weight * iou_loss


# =============================== METRICS =============================== #
def compute_metrics(pred, target, threshold=0.5, eps=1e-6):
    """
    Compute IoU, Precision, Recall, F1 for a batch of predictions.

    Args:
        pred   : (B, 1, H, W) probabilities in [0, 1]
        target : (B, 1, H, W) binary ground-truth (0 or 1)

    Returns: dict {iou, precision, recall, f1} as Python floats.
    """
    # Binarize the prediction at the given threshold
    pred_bin = (pred > threshold).float()
    target = target.float()

    # Flatten per sample for easier counting
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)                      # True Positive
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)                # False Positive
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)                # False Negative

    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


# =============================== VISUALIZATION =============================== #
def visualize_predictions(images, gt_masks, pred_masks, num_samples=4, save_path=None):
    """
    Plot input image / ground-truth mask / predicted mask / overlay side-by-side.

    Args:
        images     : tensor (B, 3, H, W) in [0, 1]
        gt_masks   : tensor (B, 1, H, W) in {0, 1}
        pred_masks : tensor (B, 1, H, W) in [0, 1]
        num_samples: how many examples to show
        save_path  : if given, save the figure to this path
    """
    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)  # keep indexing consistent

    for i in range(num_samples):
        # Move to CPU + numpy, transpose channel-last for plotting
        img_np = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        gt_np = gt_masks[i, 0].detach().cpu().numpy()
        pred_np = pred_masks[i, 0].detach().cpu().numpy()
        pred_bin = (pred_np > 0.5).astype(np.float32)

        # Build a simple red overlay: highlight predicted salient pixels
        overlay = img_np.copy()
        overlay[..., 0] = np.where(pred_bin > 0.5, 1.0, overlay[..., 0])  # red ↑
        overlay[..., 1] = np.where(pred_bin > 0.5, overlay[..., 1] * 0.5, overlay[..., 1])
        overlay[..., 2] = np.where(pred_bin > 0.5, overlay[..., 2] * 0.5, overlay[..., 2])

        axes[i, 0].imshow(img_np);    axes[i, 0].set_title("Input");          axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_np, cmap="gray");   axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_np, cmap="gray"); axes[i, 2].set_title("Prediction");   axes[i, 2].axis("off")
        axes[i, 3].imshow(overlay);   axes[i, 3].set_title("Overlay");        axes[i, 3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    plt.show()


# =============================== CHECKPOINTS =============================== #
def save_checkpoint(state, path):
    """Save a checkpoint dict (model + optimizer + epoch + best metric)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Load a checkpoint. Returns (start_epoch, best_val_loss).
    Useful for resuming training (Bonus task).
    """
    if not os.path.exists(path):
        return 0, float("inf")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"Resumed from {path} | epoch={start_epoch} | best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss