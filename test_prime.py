import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.models import PRIME
from utils.helpers import load_config, build_graph_dataloaders
from train_prime import set_seed
from utils.hierarchical_graph import *
from utils.label_formats import load_ion_positions

from torchmetrics.classification import (
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
    BinaryAUROC,
    BinarySpecificity,
)

# ======================================================
# Test (supports single model or a k-fold ensemble)
# ======================================================

@torch.no_grad()
def test_model(models, loader, device, threshold=0.5):
    """models: a list of one or more trained PRIME models. When more than
    one is given (k-fold checkpoints), each model's predicted probabilities
    are averaged per-residue before computing metrics -- matching
    DeepProSite's protocol: "we used all five trained models... for making
    inferences, and the average of these predictions was taken as the final
    prediction value."
    """
    for m in models:
        m.eval()

    auroc_metric = BinaryAUROC().to(device)

    f1_metric          = BinaryF1Score(threshold=threshold).to(device)
    mcc_metric         = BinaryMatthewsCorrCoef(threshold=threshold).to(device)
    precision_metric   = BinaryPrecision(threshold=threshold).to(device)
    recall_metric      = BinaryRecall(threshold=threshold).to(device)
    specificity_metric = BinarySpecificity(threshold=threshold).to(device)
    accuracy_metric    = BinaryAccuracy(threshold=threshold).to(device)

    n_skipped = 0

    for batch in tqdm(loader, desc="Testing", leave=False):
        for sample in batch:
            graph = sample["graph"]

            if graph.residue_labels is None:
                n_skipped += 1
                continue

            labels = graph.residue_labels.to(device).long()

            if labels.numel() == 0:
                continue

            # Run every model on this graph, average their predicted
            # probabilities. With a single model, this is a no-op.
            all_probs = []
            for m in models:
                logits = m(graph)
                probs  = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.append(probs)
            probs = torch.stack(all_probs, dim=0).mean(dim=0)
            auroc_metric.update(probs.cpu(), labels.cpu())

            f1_metric.update(probs, labels)
            mcc_metric.update(probs, labels)
            precision_metric.update(probs, labels)
            recall_metric.update(probs, labels)
            specificity_metric.update(probs, labels)
            accuracy_metric.update(probs, labels)

    if n_skipped:
        print(f"[WARN] skipped {n_skipped} sample(s) with no residue labels "
              f"(sequence/structure length mismatch)")

    results = {
        "auroc":       auroc_metric.compute().item(),
        "f1":          f1_metric.compute().item(),
        "mcc":         mcc_metric.compute().item(),
        "precision":   precision_metric.compute().item(),
        "recall":      recall_metric.compute().item(),
        "specificity": specificity_metric.compute().item(),
        "accuracy":    accuracy_metric.compute().item(),
    }

    print(f"\n{'='*50}")
    print(f"Ranking Metrics:")
    print(f"  AUROC:       {results['auroc']:.4f}")

    print(f"\nThreshold Metrics (threshold={threshold}):")
    print(f"  F1:          {results['f1']:.4f}")
    print(f"  MCC:         {results['mcc']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"{'='*50}")

    return results


def build_model(model_config, args, num_classes):
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
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True,
                         help="Task name matching a key under 'tasks:' in data_config.yaml "
                              "and under 'head:' in model_config.yaml "
                              "(e.g. Protein, Peptide, DNA, RNA, Metal_CA)")
    parser.add_argument("--data_config",  type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ckpt_path",    type=str, default=None,
                        help="Path to a single checkpoint -- if not set, auto-resolved "
                             "from task/seed/levels. Ignored if --k_folds > 1.")
    parser.add_argument("--k_folds", type=int, default=1,
                         help="If > 1, load all k fold checkpoints "
                              "(best_prime_..._fold0.pt through fold{k-1}.pt) and "
                              "average their predictions (DeepProSite-style ensemble) "
                              "instead of using a single checkpoint.")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--seed",         type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for F1/MCC/etc")
    parser.add_argument(
        "--active_levels",
        nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"]
    )
    parser.add_argument("--readout_level", type=str, default="residue")
    parser.add_argument("--test_name", type=str, default=None,
                        help="If the task's 'splits.test' in data_config.yaml is a dict of "
                             "named test sets (e.g. {129: ..., 181: ...}), select which one "
                             "to evaluate on. Ignored if 'splits.test' is a single path.")
    parser.add_argument("--test_split", type=str, default=None,
                        help="Explicit path to a test split file, overriding whatever "
                             "'splits.test' resolves to in data_config.yaml entirely.")

    args   = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load configs
    # --------------------------------------------------
    data_config  = load_config(args.data_config)
    model_config = load_config(args.model_config)

    task_cfg    = data_config["tasks"][args.task]
    num_classes = task_cfg["num_classes"]
    graph_dir   = task_cfg["graph_dir"]

    label_positions = None
    if "label_files" in task_cfg:
        label_positions = load_ion_positions(task_cfg["label_files"])
        print(f"Loaded on-the-fly label positions for {len(label_positions)} "
              f"record(s) from {task_cfg['label_files']}")

    # --------------------------------------------------
    # Resolve test split (named / explicit / single-path)
    # --------------------------------------------------
    if args.test_split is not None:
        test_split = args.test_split
    else:
        test_cfg = task_cfg["splits"]["test"]

        if isinstance(test_cfg, dict):
            if args.test_name is None:
                raise ValueError(
                    f"Task '{args.task}' has multiple named test sets in "
                    f"data_config.yaml: {list(test_cfg.keys())}. "
                    f"Pass --test_name to select one, or --test_split to "
                    f"point at a specific file directly."
                )
            test_cfg_str_keys = {str(k): v for k, v in test_cfg.items()}
            if str(args.test_name) not in test_cfg_str_keys:
                raise ValueError(
                    f"--test_name '{args.test_name}' not found for task "
                    f"'{args.task}'. Available: {list(test_cfg.keys())}"
                )
            test_split = test_cfg_str_keys[str(args.test_name)]
        else:
            test_split = test_cfg

    # --------------------------------------------------
    # Resolve checkpoint(s)
    # --------------------------------------------------
    level_tag = "_".join(args.active_levels)
    seed_tag  = f"seed{args.seed}"

    ckpt_paths = []
    if args.k_folds > 1:
        for fold_idx in range(args.k_folds):
            fold_ckpt = f"./ckpts/best_prime_{args.task}_{level_tag}_{seed_tag}_fold{fold_idx}.pt"
            if not os.path.exists(fold_ckpt):
                raise ValueError(f"Fold checkpoint not found: {fold_ckpt}")
            ckpt_paths.append(fold_ckpt)
        print(f"Ensembling {len(ckpt_paths)} fold checkpoints:")
        for p in ckpt_paths:
            print(f"  {p}")
    else:
        if args.ckpt_path is not None:
            ckpt_path = args.ckpt_path
        else:
            ckpt_path = f"./ckpts/best_prime_{args.task}_{level_tag}_{seed_tag}.pt"
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        ckpt_paths = [ckpt_path]

    print("=" * 50)
    print(f"Task:          {args.task}")
    print(f"Test split:    {test_split}"
          + (f" (name: {args.test_name})" if args.test_name else ""))
    print(f"Active levels: {args.active_levels}")
    print(f"Readout level: {args.readout_level}")
    print(f"Seed:          {args.seed}")
    print(f"K-folds:       {args.k_folds}")
    print(f"Checkpoint(s): {ckpt_paths}")
    print("=" * 50)

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------
    test_loader = build_graph_dataloaders(
        graph_dir=graph_dir,
        test_split=test_split,
        batch_size=args.batch_size,
        device=device,
        test_only=True,
        label_positions=label_positions,
    )

    # --------------------------------------------------
    # Model(s)
    # --------------------------------------------------
    models = []
    for ckpt_path in ckpt_paths:
        model = build_model(model_config, args, num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()
        models.append(model)

    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    results = test_model(models, test_loader, device, threshold=args.threshold)