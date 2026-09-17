import sys
import os
import glob
import gc
import csv
import torch

from transformers import EsmFold2Model

MODEL_NAME = "biohub/ESMFold2-hf"


def parse_label_file(path):
    """Parse '>header / sequence / 0101...labels' triplets into {id: sequence}."""
    sequences = {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip() != ""]

    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.startswith(">"):
            i += 1
            continue
        record_id = header[1:].strip()
        if i + 1 < len(lines):
            sequences[record_id] = lines[i + 1].strip()
        i += 3

    return sequences


def collect_sequences(task_dir):
    """Find every .txt / .fa raw label file anywhere under task_dir (including
    nested subfolders for tasks with multiple train/test benchmark groups,
    e.g. PPI-site/Train352-Test70/) and parse all id->sequence pairs from them.
    Explicitly skips anything under a 'splits/' or 'PDBs/' directory, since
    those hold derived split-id lists and folded structures, not raw label data.
    """
    label_files = sorted(
        glob.glob(os.path.join(task_dir, "**", "*.txt"), recursive=True) +
        glob.glob(os.path.join(task_dir, "**", "*.fa"), recursive=True)
    )
    label_files = [
        p for p in label_files
        if os.sep + "splits" + os.sep not in p
        and os.sep + "PDBs" + os.sep not in p
        and not os.path.basename(p) == "failed_sequences.csv"
    ]

    print(f"Found {len(label_files)} raw label file(s) under {task_dir}: "
          f"{[os.path.relpath(f, task_dir) for f in label_files]}")

    sequences = {}
    for path in label_files:
        parsed = parse_label_file(path)
        print(f"  {os.path.relpath(path, task_dir)}: {len(parsed)} sequences")
        sequences.update(parsed)

    return sequences


def collect_split_ids(task_dir):
    """Read every train_split_*.txt and test_split_*.txt under task_dir/splits/
    and return the union of record ids across all of them -- this is the
    authoritative list of exactly which proteins are actually needed for
    folding, across every train/test sub-benchmark this task defines."""
    splits_dir = os.path.join(task_dir, "splits")
    split_files = sorted(
        glob.glob(os.path.join(splits_dir, "train_split_*.txt")) +
        glob.glob(os.path.join(splits_dir, "test_split_*.txt"))
    )

    if not split_files:
        raise ValueError(
            f"No train_split_*.txt / test_split_*.txt files found under "
            f"{splits_dir} -- run make_splits.py for this task first."
        )

    print(f"Found {len(split_files)} split file(s) in {splits_dir}: "
          f"{[os.path.basename(f) for f in split_files]}")

    ids = set()
    for path in split_files:
        with open(path) as f:
            file_ids = [line.strip() for line in f if line.strip()]
        print(f"  {os.path.basename(path)}: {len(file_ids)} ids")
        ids.update(file_ids)

    print(f"Total unique ids needed across all splits: {len(ids)}")
    return ids


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    model = EsmFold2Model.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0},
        use_kernels=False,
    )
    return model


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def try_fold(model, seq, num_diffusion_samples):
    samples_to_try = num_diffusion_samples
    while samples_to_try >= 1:
        try:
            with torch.no_grad():
                pdb_str = model.infer_protein_as_pdb(seq, num_diffusion_samples=samples_to_try)
            return pdb_str, samples_to_try
        except torch.cuda.OutOfMemoryError:
            free_gpu_memory()
            if samples_to_try == 1:
                return None, samples_to_try
            samples_to_try = max(1, samples_to_try // 2)
            print(f"    OOM at num_diffusion_samples={samples_to_try * 2}, "
                  f"retrying with num_diffusion_samples={samples_to_try} ...")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu_memory()
                if samples_to_try == 1:
                    return None, samples_to_try
                samples_to_try = max(1, samples_to_try // 2)
                print(f"    OOM at num_diffusion_samples={samples_to_try * 2}, "
                      f"retrying with num_diffusion_samples={samples_to_try} ...")
            else:
                raise

    return None, samples_to_try


def fold_and_save(model, sequences, outdir, num_diffusion_samples=8,
                   max_length=None, failed_manifest_path=None):
    os.makedirs(outdir, exist_ok=True)
    ordered = sorted(sequences.items(), key=lambda kv: len(kv[1]))

    failures = []
    n = len(ordered)

    for i, (record_id, seq) in enumerate(ordered, 1):
        out_path = os.path.join(outdir, f"{record_id}.pdb")

        if os.path.exists(out_path):
            print(f"[{i}/{n}] {record_id}: already exists, skipping")
            continue
        if not seq:
            print(f"[{i}/{n}] {record_id}: empty sequence, skipping")
            continue
        if max_length is not None and len(seq) > max_length:
            print(f"[{i}/{n}] {record_id}: length {len(seq)} > max_length "
                  f"{max_length}, skipping")
            failures.append((record_id, len(seq), "too_long_skipped"))
            continue

        try:
            pdb_str, used_samples = try_fold(model, seq, num_diffusion_samples)
        except Exception as e:
            # Catch-all so one unexpected non-OOM error (e.g. a malformed
            # sequence) also doesn't kill the whole job -- log and move on.
            print(f"[{i}/{n}] {record_id}: FAILED with non-OOM error: {e}")
            failures.append((record_id, len(seq), f"error: {e}"))
            free_gpu_memory()
            continue

        if pdb_str is None:
            print(f"[{i}/{n}] {record_id}: OOM even at num_diffusion_samples=1 "
                  f"(length {len(seq)}), skipping")
            failures.append((record_id, len(seq), "oom_at_min_samples"))
            continue

        with open(out_path, "w") as f:
            f.write(pdb_str)

        note = "" if used_samples == num_diffusion_samples else f" [reduced to {used_samples} samples due to OOM]"
        print(f"[{i}/{n}] {record_id}: saved ({len(seq)} residues){note}")
        free_gpu_memory()

    if failed_manifest_path and failures:
        with open(failed_manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id", "length", "reason"])
            writer.writerows(failures)
        print(f"\n{len(failures)} sequence(s) failed/skipped -- see {failed_manifest_path}")


def main():
    task_dir = sys.argv[1].rstrip("/")
    outdir = os.path.join(task_dir, "PDBs")
    failed_manifest_path = os.path.join(task_dir, "failed_sequences.csv")

    # 1. Sequence source: recursively scan every raw label file under
    #    task_dir (handles nested subfolders for multi-split tasks).
    all_sequences = collect_sequences(task_dir)
    print(f"Total sequences found in raw label files: {len(all_sequences)}")

    # 2. Fold list: the authoritative set of ids actually needed, taken
    #    from splits/train_split_*.txt + splits/test_split_*.txt.
    needed_ids = collect_split_ids(task_dir)

    sequences = {}
    missing_ids = []
    for rid in needed_ids:
        if rid in all_sequences:
            sequences[rid] = all_sequences[rid]
        else:
            missing_ids.append(rid)

    print(f"\n{len(sequences)}/{len(needed_ids)} split ids matched to a "
          f"sequence from raw label files.")
    if missing_ids:
        print(f"[WARN] {len(missing_ids)} split id(s) have NO matching "
              f"sequence in any raw label file under {task_dir} -- these "
              f"will be skipped: {missing_ids[:10]}"
              f"{' ...' if len(missing_ids) > 10 else ''}")

    model = load_model()
    fold_and_save(
        model,
        sequences,
        outdir,
        num_diffusion_samples=8,
        max_length=None,
        failed_manifest_path=failed_manifest_path,
    )

    print("Done.")


if __name__ == "__main__":
    main()