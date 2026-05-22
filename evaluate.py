"""
evaluate.py
-----------
Evaluate the trained Salient Object Detection model on the test set.

Steps:
1. Load the best model from checkpoints/best_model.pt
2. Use test_loader from data_loader.py
3. Run inference on every test batch
4. Compute IoU, Precision, Recall, F1 (averaged over the whole test set)
5. Print the final average metrics
6. Generate sample visualizations:
       input image | ground-truth mask | predicted mask | overlay
   We save three figures: best predictions, worst predictions, and random
   samples — so you can see both successes and failure cases.

Usage:
    python evaluate.py
    python evaluate.py --num_visualize 6 --visualization_dir results/
"""

import os
import argparse
import random

import torch
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model import SODModel
from utils import compute_metrics, visualize_predictions


@torch.no_grad()
def evaluate(model, loader, device, collect_samples=True):
    """
    Run the model on the loader and return:
      - average IoU / Precision / Recall / F1 across all test samples
      - (optional) a list of per-sample records for visualization, each
        containing the image, ground-truth mask, prediction, and per-sample IoU

    We compute metrics per-batch (using utils.compute_metrics) and then
    take a sample-weighted average so batches of different sizes are
    handled correctly.
    """
    model.eval()

    # Sums of (metric * batch_size) so we can divide by total samples at the end
    metric_sums = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_samples = 0

    # Per-sample records used to choose best / worst / random examples to plot.
    # Each entry is (image_tensor, gt_tensor, pred_tensor, sample_iou).
    samples = []

    for images, masks in tqdm(loader, desc="Evaluating on test set"):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass -> predicted saliency probabilities in [0, 1]
        preds = model(images)

        # Per-batch metrics (already averaged over the batch internally)
        batch_metrics = compute_metrics(preds, masks)

        batch_size = images.size(0)
        for k in metric_sums:
            metric_sums[k] += batch_metrics[k] * batch_size
        total_samples += batch_size

        # ---- Collect per-sample records for visualization ---- #
        if collect_samples:
            # Compute a per-sample (binarized) IoU so we can rank examples
            preds_bin = (preds > 0.5).float()
            preds_flat = preds_bin.view(batch_size, -1)
            masks_flat = masks.view(batch_size, -1)
            inter = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - inter
            sample_iou = (inter + 1e-6) / (union + 1e-6)   # shape: (B,)

            # Move to CPU once, then split into per-sample tensors to keep
            # memory low and avoid holding GPU memory between batches.
            imgs_cpu = images.detach().cpu()
            gts_cpu = masks.detach().cpu()
            prs_cpu = preds.detach().cpu()
            ious_cpu = sample_iou.detach().cpu().tolist()

            for i in range(batch_size):
                samples.append((imgs_cpu[i], gts_cpu[i], prs_cpu[i], ious_cpu[i]))

    # Final averages over the entire test set
    avg = {k: v / max(total_samples, 1) for k, v in metric_sums.items()}
    return avg, total_samples, samples


def _stack_samples(records):
    """Turn a list of (img, gt, pred, iou) records into batched tensors."""
    imgs = torch.stack([r[0] for r in records], dim=0)
    gts = torch.stack([r[1] for r in records], dim=0)
    prs = torch.stack([r[2] for r in records], dim=0)
    return imgs, gts, prs


def visualize_results(samples, num_visualize, visualization_dir, seed=42):
    """
    Save three figures showing input | ground-truth | prediction | overlay:

      1. best_predictions.png   -> highest-IoU samples (model successes)
      2. worst_predictions.png  -> lowest-IoU samples  (model failures)
      3. random_predictions.png -> a random selection  (unbiased view)

    Each figure shows up to `num_visualize` rows.
    """
    if not samples:
        print("No samples available for visualization. Skipping.")
        return

    os.makedirs(visualization_dir, exist_ok=True)
    k = min(num_visualize, len(samples))

    # Sort samples by per-sample IoU (descending = best first)
    sorted_samples = sorted(samples, key=lambda r: r[3], reverse=True)

    # ---- Best (top-k by IoU) ---- #
    best = sorted_samples[:k]
    best_imgs, best_gts, best_prs = _stack_samples(best)
    best_path = os.path.join(visualization_dir, "best_predictions.png")
    print(f"\n[viz] Best predictions  (IoUs: {[round(r[3], 3) for r in best]})")
    visualize_predictions(best_imgs, best_gts, best_prs,
                          num_samples=k, save_path=best_path)

    # ---- Worst (bottom-k by IoU) ---- #
    worst = sorted_samples[-k:][::-1]   # reverse so the very worst is first
    worst_imgs, worst_gts, worst_prs = _stack_samples(worst)
    worst_path = os.path.join(visualization_dir, "worst_predictions.png")
    print(f"[viz] Worst predictions (IoUs: {[round(r[3], 3) for r in worst]})")
    visualize_predictions(worst_imgs, worst_gts, worst_prs,
                          num_samples=k, save_path=worst_path)

    # ---- Random k samples ---- #
    rng = random.Random(seed)
    rand = rng.sample(samples, k)
    rand_imgs, rand_gts, rand_prs = _stack_samples(rand)
    rand_path = os.path.join(visualization_dir, "random_predictions.png")
    print(f"[viz] Random predictions (IoUs: {[round(r[3], 3) for r in rand]})")
    visualize_predictions(rand_imgs, rand_gts, rand_prs,
                          num_samples=k, save_path=rand_path)

    print(f"\nVisualizations saved to: {os.path.abspath(visualization_dir)}")


def main(args):
    # ---- Device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Test data ---- #
    # We only need test_loader, but get_dataloaders returns all three.
    _, _, test_loader = get_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ---- Load model + best checkpoint ---- #
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            f"Please train the model first by running train.py."
        )

    model = SODModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Support both checkpoint formats: {"model_state_dict": ...} or a raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_info = ckpt.get("epoch", "?")
    else:
        model.load_state_dict(ckpt)
        epoch_info = "?"
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {epoch_info})")

    # ---- Run evaluation ---- #
    metrics, n_samples, samples = evaluate(
        model, test_loader, device, collect_samples=True
    )

    # ---- Print final results ---- #
    print("\n=========== Test Set Results ===========")
    print(f"Samples evaluated : {n_samples}")
    print(f"IoU       : {metrics['iou']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print("========================================\n")

    # ---- Visualize sample predictions ---- #
    visualize_results(
        samples=samples,
        num_visualize=args.num_visualize,
        visualization_dir=args.visualization_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--masks_dir", type=str, default="dataset/masks")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--num_visualize", type=int, default=4,
                        help="How many samples to show per figure (best/worst/random).")
    parser.add_argument("--visualization_dir", type=str, default="results",
                        help="Folder where prediction visualizations are saved.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)