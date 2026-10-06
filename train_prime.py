import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from models.models import PRIME
from torchmetrics.classification import BinaryF1Score, BinaryAveragePrecision
import argparse
from tqdm import tqdm
import random
import numpy as np
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hierarchical_graph import *
from utils.helpers import compute_class_weights, load_config, build_graph_dataloaders, CombinedLoss, set_seed
from utils.dataset import HierarchicalGraphDataset
from utils.label_formats import load_ion_positions

# ======================================================
# Metrics
# ======================================================

def get_f1_metric(device):
    return BinaryF1Score(threshold=0.5).to(device)


def get_auprc_metric(device):
    return BinaryAveragePrecision().to(device)

# ======================================================
# Train one epoch
# ======================================================

def add_gaussian_noise(x, eps):
    if eps <= 0:
        return x
    return x + torch.randn_like(x) * eps


def train_one_epoch(
    model, loader, optimizer, criterion,
    device, f1_metric, auprc_metric, grad_clip=1.0, augment_eps=0.0,
):
    model.train()
    f1_metric.reset()
    auprc_metric.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.tensor(0.0, device=device)
        n_valid_samples = 0

        for sample in batch:
            graph = sample["graph"]

            if graph.residue_labels is None:
                continue

            labels = graph.residue_labels.to(device).long()
            if augment_eps > 0:
                graph.residue.X = add_gaussian_noise(graph.residue.X, augment_eps)

            logits = model(graph)

            if labels.numel() == 0:
                continue

            loss        = criterion(logits, labels)
            batch_loss += loss
            n_valid_samples += 1

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu()
                f1_metric.update(probs, labels.cpu())
                auprc_metric.update(probs, labels.cpu().int())

        if n_valid_samples == 0:
            continue

        (batch_loss / n_valid_samples).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += (batch_loss / n_valid_samples).item()

    return total_loss / len(loader), f1_metric.compute().item(), auprc_metric.compute().item()

# ======================================================
# Evaluate
# ======================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, f1_metric, auprc_metric):
    model.eval()
    f1_metric.reset()
    auprc_metric.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch_loss = 0.0
        n_valid_samples = 0

        for sample in batch:
            graph = sample["graph"]

            if graph.residue_labels is None:
                continue

            labels = graph.residue_labels.to(device).long()

            logits = model(graph)

            if labels.numel() == 0:
                continue

            loss        = criterion(logits, labels)
            batch_loss += loss.item()
            n_valid_samples += 1

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu()
            f1_metric.update(probs, labels.cpu())
            auprc_metric.update(probs, labels.cpu().int())

        if n_valid_samples == 0:
            continue

        total_loss += batch_loss / n_valid_samples

    return total_loss / len(loader), f1_metric.compute().item(), auprc_metric.compute().item()

# ======================================================
# Train
# ======================================================

def train_prime(
    model,
    train_loader,
    val_loader,
    criterion,
    output_path,
    device,
    task_name="Peptide",
    log_path="training_log.txt",
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    patience_es=20,
    factor=0.6,
    patience_lr=5,
    grad_clip=5.0,
    warmup_epochs=3,
    augment_eps=0.0,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience_lr
    )

    train_f1_metric,    train_auprc_metric = get_f1_metric(device), get_auprc_metric(device)
    val_f1_metric,      val_auprc_metric   = (get_f1_metric(device), get_auprc_metric(device)) if val_loader is not None else (None, None)
    use_val_for_selection = val_loader is not None
    best_score          = -float("inf")
    early_stop_counter  = 0

    with open(log_path, "w") as log_file:
        log_file.write(f"Task:        {task_name} (binary)\n")
        log_file.write(f"LR:          {lr} | WD: {weight_decay}\n")
        log_file.write(f"Epochs:      {epochs}\n")
        log_file.write(f"Metric:      AUPRC (model selection) + BinaryF1 @ threshold=0.5 (logged)\n")
        log_file.write(f"Val loader:  {'None' if val_loader is None else 'provided'}\n")
        log_file.write(f"Model selection / early stopping based on: "
                        f"{'Val AUPRC' if use_val_for_selection else 'Train AUPRC'}\n")
        log_file.write("=" * 50 + "\n")
        log_file.flush()

        for epoch in range(epochs):

            train_loss, train_f1, train_auprc = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, train_f1_metric, train_auprc_metric, grad_clip,
                augment_eps=augment_eps,
            )

            if val_loader is not None:
                val_loss, val_f1, val_auprc = evaluate(
                    model, val_loader, criterion,
                    device, val_f1_metric, val_auprc_metric,
                )
                val_str = (f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                           f"Val AUPRC: {val_auprc:.4f} | ")
                log_file.write(f"Val   Loss: {val_loss:.6f} | Val   F1: {val_f1:.6f} | "
                               f"Val   AUPRC: {val_auprc:.6f}\n")
            else:
                val_str = ""

            if epoch < warmup_epochs:
                warmup.step()
            else:
                plateau.step(val_loss if use_val_for_selection else train_loss)

            current_lr = optimizer.param_groups[0]['lr']

            log_file.write(f"Epoch {epoch+1:03d}\n")
            log_file.write(f"Train Loss: {train_loss:.6f} | Train F1: {train_f1:.6f} | "
                           f"Train AUPRC: {train_auprc:.6f}\n")
            log_file.write(f"LR: {current_lr:.2e}\n")
            log_file.write("-" * 40 + "\n")
            log_file.flush()

            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                  f"Train AUPRC: {train_auprc:.4f} | "
                  f"{val_str}"
                  f"LR: {current_lr:.2e}")

            current_score = val_auprc if use_val_for_selection else train_auprc

            if current_score > best_score:
                best_score          = current_score
                early_stop_counter  = 0
                torch.save(model.state_dict(), output_path)
                log_file.write("New best model saved.\n")
                print("Best model saved")
            else:
                early_stop_counter += 1
                log_file.write(f"EarlyStopping: {early_stop_counter}/{patience_es}\n")
                if early_stop_counter >= patience_es:
                    log_file.write("Early stopping triggered.\n")
                    print("Early stopping triggered.")
                    break

        log_file.write(f"Best {'Val' if use_val_for_selection else 'Train'} AUPRC: {best_score:.6f}\n")
        log_file.write("Training Finished\n")

    return best_score

# ======================================================
# K-fold split (in-memory, fixed seed, no files touched
# outside a temp dir -- make_splits.py is untouched)
# ======================================================

def build_criterion(loss_name, class_weights):
    if loss_name == "combined":
        return CombinedLoss(class_weights=class_weights, alpha=0.5)
    elif loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown --loss: {loss_name}")


def read_ids(path):
    with open(path) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def make_kfold_splits(train_split_path, k, seed):
    """Shuffle the full training id list with a fixed seed and partition
    into k roughly-equal folds. Returns a list of (train_ids, val_ids)
    tuples, one per fold."""
    ids = read_ids(train_split_path)
    rng = random.Random(seed)
    rng.shuffle(ids)

    fold_size = len(ids) // k
    folds = [ids[i*fold_size:(i+1)*fold_size] for i in range(k - 1)]
    folds.append(ids[(k - 1) * fold_size:])  # last fold takes the remainder

    splits = []
    for i, val_ids in enumerate(folds):
        train_ids = [rid for j, fold in enumerate(folds) if j != i for rid in fold]
        splits.append((train_ids, val_ids))
    return splits


def write_temp_split(ids, tmp_dir, name):
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        for rid in ids:
            f.write(f"{rid}\n")
    return path


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                         help="Task name matching a key under 'tasks:' in data_config.yaml. "
                              "For Metal, use 'Metal_CA', 'Metal_FE', etc. to select a "
                              "specific ion sub-task.")
    parser.add_argument("--data_config",  type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",    type=float, default=5.0)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=1)
    parser.add_argument(
        "--active_levels",
        nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"],
    )
    parser.add_argument("--readout_level", type=str, default="residue")
    parser.add_argument("--patience_es", type=int, default=15,
                         help="Early stopping patience (epochs with no Val F1 improvement)")
    parser.add_argument("--patience_lr", type=int, default=5,
                         help="LR plateau scheduler patience")
    parser.add_argument("--loss", type=str, default="combined",
                         choices=["combined", "weighted_ce", "ce"],
                         help="'combined' = weighted CE + Dice (current default), "
                              "'weighted_ce' = class-weighted CE only (no Dice), "
                              "'ce' = plain unweighted CE, matching DeepPANB/DeepProSite")
    parser.add_argument("--augment_eps", type=float, default=0.0,
                         help="Gaussian noise magnitude added to residue-level "
                              "features during training (DeepProSite-style feature "
                              "augmentation; their sweep found eps=0.1 optimal). "
                              "0.0 = disabled (default).")
    parser.add_argument("--k_folds", type=int, default=1,
                         help="If > 1, run k-fold cross-validation over the training "
                              "split instead of a single train/val run. Test set (if "
                              "used later via test_prime.py) is untouched either way.")
    parser.add_argument(
        "--direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "bottom_up_only", "top_down_only"],
        help="Message passing direction in the hierarchical encoder "
             "(bidirectional=default, no suffix in ckpt/log name). Used "
             "for the bottom-up/top-down ablation study.",
    )

    args   = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load configs
    # --------------------------------------------------
    data_config  = load_config(args.data_config)
    model_config = load_config(args.model_config)
    task_cfg     = data_config["tasks"][args.task]
    num_classes  = task_cfg["num_classes"]

    label_positions = None
    if "label_files" in task_cfg:
        label_positions = load_ion_positions(task_cfg["label_files"])
        print(f"Loaded on-the-fly label positions for {len(label_positions)} "
              f"record(s) from {task_cfg['label_files']}")

    graph_dir   = task_cfg["graph_dir"]
    train_split = task_cfg["splits"]["train"]

    def build_model():
        return PRIME(
            num_classes=num_classes,
            input_dims=model_config["hierarchical"]["input_dims"],
            active_levels=args.active_levels,
            readout_level=args.readout_level,
            hidden_dim=model_config["hierarchical"]["hidden_dim"],
            encoder_layers=model_config["hierarchical"]["n_layers"],
            head_hidden_dim=model_config["head"][args.task]["hidden_dim"],
            head_layers=model_config["head"][args.task]["num_layers"],
            dropout=model_config["head"][args.task]["dropout"],
            task_level="node",
            direction=args.direction,
        ).to(device)

    level_tag     = "_".join(args.active_levels)
    seed_tag      = f"seed{args.seed}"
    direction_tag = f"_{args.direction}" if args.direction != "bidirectional" else ""
    os.makedirs("./ckpts", exist_ok=True)
    os.makedirs("./logs",  exist_ok=True)

    print("=" * 40)
    print(f"Task:          {args.task} (binary)")
    print(f"Num classes:   {num_classes}")
    print(f"Readout level: {args.readout_level}")
    print(f"Active levels: {args.active_levels}")
    print(f"Direction:     {args.direction}")
    print(f"Seed:          {args.seed}")
    print(f"K-folds:       {args.k_folds}")
    print(f"Patience (ES): {args.patience_es}")
    print(f"Patience (LR): {args.patience_lr}")
    print("=" * 40)

    if args.k_folds <= 1:
        # ---------------------------------------------
        # Original single train/val run (unchanged)
        # ---------------------------------------------
        val_split = task_cfg["splits"].get("val", None)

        train_loader, val_loader = build_graph_dataloaders(
            graph_dir=graph_dir,
            train_split=train_split,
            val_split=val_split,
            batch_size=args.batch_size,
            device=device,
            label_positions=label_positions,
        )

        print("Computing class weights...")
        class_weights = compute_class_weights(train_loader, device)
        criterion = build_criterion(args.loss, class_weights)

        model = build_model()
        if args.resume is not None:
            if not os.path.exists(args.resume):
                raise ValueError(f"Checkpoint not found: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"Resumed from: {args.resume}")

        output_path = f"./ckpts/best_prime_{args.task}_{level_tag}{direction_tag}_{seed_tag}.pt"
        log_path    = f"./logs/training_log_prime_{args.task}_{level_tag}{direction_tag}_{seed_tag}.txt"

        train_prime(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, output_path=output_path, log_path=log_path,
            task_name=args.task, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            patience_es=args.patience_es, patience_lr=args.patience_lr,
            augment_eps=args.augment_eps,
            device=device,
        )

    else:
        # ---------------------------------------------
        # K-fold cross-validation over the training split.
        # ---------------------------------------------
        splits = make_kfold_splits(train_split, k=args.k_folds, seed=args.seed)
        print(f"Generated {len(splits)} fold(s) from {train_split} "
              f"(seed={args.seed}, fixed for reproducibility)")

        fold_scores = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(splits):
                print(f"\n{'='*40}\nFold {fold_idx+1}/{args.k_folds} "
                      f"({len(fold_train_ids)} train / {len(fold_val_ids)} val)\n{'='*40}")

                fold_train_path = write_temp_split(fold_train_ids, tmp_dir, f"train_fold{fold_idx}.txt")
                fold_val_path   = write_temp_split(fold_val_ids, tmp_dir, f"val_fold{fold_idx}.txt")

                train_loader, val_loader = build_graph_dataloaders(
                    graph_dir=graph_dir,
                    train_split=fold_train_path,
                    val_split=fold_val_path,
                    batch_size=args.batch_size,
                    device=device,
                    label_positions=label_positions,
                )

                print("Computing class weights...")
                class_weights = compute_class_weights(train_loader, device)
                criterion = build_criterion(args.loss, class_weights)

                model = build_model()

                fold_output_path = f"./ckpts/best_prime_{args.task}_{level_tag}{direction_tag}_{seed_tag}_fold{fold_idx}.pt"
                fold_log_path    = f"./logs/training_log_prime_{args.task}_{level_tag}{direction_tag}_{seed_tag}_fold{fold_idx}.txt"

                best_val_auprc = train_prime(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    criterion=criterion, output_path=fold_output_path, log_path=fold_log_path,
                    task_name=f"{args.task} (fold {fold_idx})", epochs=args.epochs, lr=args.lr,
                    weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                    patience_es=args.patience_es, patience_lr=args.patience_lr,
                    augment_eps=args.augment_eps,
                    device=device,
                )
                fold_scores.append(best_val_auprc)
                print(f"Fold {fold_idx+1} best Val AUPRC: {best_val_auprc:.4f}")

        mean_auprc = float(np.mean(fold_scores))
        std_auprc  = float(np.std(fold_scores))
        print(f"\n{'='*40}")
        print(f"{args.k_folds}-fold CV results for {args.task}:")
        print(f"  Per-fold Val AUPRC: {[f'{s:.4f}' for s in fold_scores]}")
        print(f"  Mean Val AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")
        print(f"{'='*40}")

        summary_path = f"./logs/kfold_summary_{args.task}_{level_tag}{direction_tag}_{seed_tag}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Task: {args.task}\n")
            f.write(f"Direction: {args.direction}\n")
            f.write(f"K-folds: {args.k_folds}, seed: {args.seed}\n")
            f.write(f"Model selection metric: Val AUPRC\n")
            f.write(f"Per-fold Val AUPRC: {fold_scores}\n")
            f.write(f"Mean Val AUPRC: {mean_auprc:.6f} +/- {std_auprc:.6f}\n")
        print(f"Summary written to {summary_path}")