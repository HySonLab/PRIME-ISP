#!/bin/bash
# predict.sh -- run PRIME-ISP on your own protein.
# Edit the variables below, then run: bash predict.sh

# ---------------- Input (set exactly one) ----------------
SEQUENCE=""                     # amino-acid sequence, e.g. "MKTAYIAKQRQISFVKSHFSRQ..."
FASTA=""                        # path to a FASTA file (one protein per record)
STRUCTURE=""                    # path to a PDB / mmCIF file
CHAINS=""                       # chains to keep from STRUCTURE, e.g. "A B" (empty = all)

# ---------------- Partner types ----------------
# any of: protein peptide dna rna metal_ca metal_fe metal_mn carbohydrate small_molecule
# or "all"
PARTNERS="all"

# ---------------- Settings ----------------
CUDA_DEVICE=0
OUTPUT_DIR="./prime_isp_output"
CKPT_DIR="./ckpts"
K_FOLDS=5
SEED=1
THRESHOLD=0.5
RENDER="auto"                   # auto | pymol | matplotlib | none

# ---------------- Run ----------------
N_INPUTS=0
for v in "$SEQUENCE" "$FASTA" "$STRUCTURE"; do
    [ -n "$v" ] && N_INPUTS=$((N_INPUTS + 1))
done
if [ "$N_INPUTS" -ne 1 ]; then
    echo "Set exactly one of SEQUENCE, FASTA, or STRUCTURE in predict.sh" >&2
    exit 1
fi

ARGS=(--partners $PARTNERS --output_dir "$OUTPUT_DIR" --ckpt_dir "$CKPT_DIR"
      --k_folds "$K_FOLDS" --seed "$SEED" --threshold "$THRESHOLD"
      --render "$RENDER")

if   [ -n "$SEQUENCE" ];  then ARGS+=(--sequence "$SEQUENCE")
elif [ -n "$FASTA" ];     then ARGS+=(--fasta "$FASTA")
else
    ARGS+=(--structure "$STRUCTURE")
    [ -n "$CHAINS" ] && ARGS+=(--chains $CHAINS)
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py "${ARGS[@]}"