import torch
from collections import Counter
import yaml
from .dataset import HierarchicalGraphDataset
from torch.utils.data import DataLoader
import numpy as np

from torchmetrics.classification import MulticlassAccuracy, MultilabelF1Score, BinaryAUROC
from torchmetrics import Metric
from torch import nn
import random

# ======================================================
# Seed
# ======================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary classification.
    Directly optimizes overlap between predicted and ground truth pockets.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # ignore index -1
        valid_mask = targets >= 0
        logits     = logits[valid_mask]
        targets    = targets[valid_mask].float()

        probs = torch.softmax(logits, dim=-1)[:, 1]  # pocket probability

        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class F1Max(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.detach())
        self.targets.append(targets.float().detach())

    def compute(self) -> torch.Tensor:
        preds   = torch.cat(self.preds,   dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.f1_max(preds, targets)

    def f1_max(self, pred, target):
        pred = torch.softmax(pred, dim=1)  
        if target.ndim == 1:
            target = F.one_hot(
                target.long(), num_classes=pred.shape[1]
            ).float()
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order).int()
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)
        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = (
            2
            * all_precision
            * all_recall
            / (all_precision + all_recall + 1e-10)
        )
        return all_f1.max()

def add_noise(x, noise_std=None):
    """
    Stable noise injection for denoising AE.
    Wider noise range gives the model a real learning signal.
    """
    x_clean = x.clone()

    if noise_std is None:
        noise_std = torch.empty(1).uniform_(0.1, 0.5).item()

    noise = torch.randn_like(x_clean) * noise_std
    
    x_noisy = x_clean + noise

    return x_noisy, x_clean, noise_std

def normalize_coords(x, mode="std", eps=1e-6):
    """
    Normalize 3D coordinates.

    Args:
        x: (N, 3) tensor
        mode: "std" or "radius"
        eps: numerical stability

    Returns:
        normalized x
    """

    # Center
    x = x - x.mean(dim=0, keepdim=True)

    if mode == "std":
        std = x.std(dim=0, keepdim=True)
        std[std < eps] = 1.0
        x = x / std

    elif mode == "radius":
        radius = torch.norm(x, dim=1).max()
        x = x / (radius + eps)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x

def get_metric(task_type, num_classes, device):
    if task_type == "multilabel_classification":
        return F1Max().to(device)
    elif task_type == "node_classification":
        return BinaryAUROC().to(device)
    else:
        return MulticlassAccuracy(
            num_classes=num_classes
        ).to(device)

# ======================================================
# Class weights (binary)
# ======================================================

def compute_class_weights(loader, device):
    """Compute binary class weights from training data."""
    counts = torch.zeros(2)

    for batch in loader:
        for sample in batch:
            graph = sample["graph"]

            if graph.residue_labels is None:
                continue

            labels = graph.residue_labels.cpu()
            counts[0] += (labels == 0).sum()
            counts[1] += (labels == 1).sum()

    total   = counts.sum()
    weights = total / (counts * 2 + 1e-6)
    weights[counts == 0] = 0.0

    # cap extreme weights
    max_weight = weights[weights > 0].median() * 10
    weights    = torch.clamp(weights, max=max_weight)
    weights    = weights / weights.sum() * 2

    print(f"Binary class counts:  neg={counts[0].long()}, pos={counts[1].long()}")
    print(f"Binary class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")

    return weights.to(device)

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5):
        super().__init__()
        self.dice  = BinaryDiceLoss()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.alpha = alpha   # weight between dice and ce

    def forward(self, logits, targets):
        return self.alpha * self.dice(logits, targets) + \
               (1 - self.alpha) * self.ce(logits, targets)

def compute_pos_weight(loader, num_classes):

    pos_counts = torch.zeros(num_classes)
    total_samples = 0

    for batch in loader:
        for sample in batch:
            y = sample["label"]
            for cls in y:
                pos_counts[cls] += 1
            total_samples += 1

    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    return pos_weight

def to_multihot(y, num_classes, device=None):
    """
    Convert a tensor/list of class indices to a binary multi-hot vector.
    
    Args:
        y:           tensor or list of local class indices
        num_classes: total number of classes
        device:      target device
    
    Returns:
        binary vector of shape (num_classes,)
    """
    if isinstance(y, torch.Tensor):
        y = y.long().tolist()
    elif isinstance(y, (int, float)):
        y = [int(y)]

    # validate range
    y = [cls for cls in y if 0 <= cls < num_classes]

    vec = torch.zeros(num_classes)
    if len(y) > 0:
        vec[y] = 1.0

    return vec.to(device) if device is not None else vec


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_graph_dataloaders(
    graph_dir:   str,
    train_split: str = None,
    val_split:   str  = None,
    test_split:  str  = None,
    batch_size:  int  = 4,
    num_workers: int  = 0,
    device            = None,
    test_only:   bool = False,
    label_positions:   dict = None,
):
    if test_only:
        if test_split is None:
            raise ValueError("test_split required for test_only mode")

        test_dataset = HierarchicalGraphDataset(
            graph_dir=graph_dir,
            split_file=test_split,
            device=device,
            label_positions=label_positions,
        )
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )

    train_dataset = HierarchicalGraphDataset(
        graph_dir=graph_dir,
        split_file=train_split,
        device=device,
        label_positions=label_positions,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )

    if val_split is not None:
        val_dataset = HierarchicalGraphDataset(
            graph_dir=graph_dir,
            split_file=val_split,
            device=device,
            label_positions=label_positions,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )
        return train_loader, val_loader

    return train_loader, None