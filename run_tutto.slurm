#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline, in sequence, for each heatmap mode:
#   unit tests → Stage 1 → (landmark_stats) → Stage 2 → evaluation + calibration
#
# Submit from the repository root (the logs/ folder must exist):
#   mkdir -p logs && sbatch run_tutto.slurm
# Only some modes / other settings:
#   sbatch --export=ALL,MODES="fixed" run_tutto.slurm
#   sbatch --export=ALL,MODES="learned_sigma probabilistic",BATCH_HEATMAP=8 run_tutto.slurm
# Quick test of the whole sequence (2 epochs per training):
#   sbatch --export=ALL,EPOCHS=2,OUT=esperimenti_prova --time=02:00:00 run_tutto.slurm
#
# Results:  esperimenti/<mode>/
#   modello_heatmap.pth, modello_refine.pth, landmark_stats.py,
#   calibrazione_incertezza.json, 1_train_heatmap.log, 2_train_refine.log, 3_evaluate.log
#
# Do NOT run two of these jobs at the same time in the same folder: they share
# src/landmark_stats.py.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ceph_tutto
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_H100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G 
#SBATCH --time=24:00:00


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# ── Python environment ───────────────────────────────────────────────────────
# Conda env named "LLM" by default (CONDA_ENV=none: use the python already in PATH;
# with a venv, replace this block with: source <venv>/bin/activate)
CONDA_ENV="${CONDA_ENV:-LLM}"
if [ "$CONDA_ENV" != "none" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

MODES="${MODES:-fixed learned_sigma probabilistic}"
BATCH_HEATMAP="${BATCH_HEATMAP:-16}"
BATCH_REFINE="${BATCH_REFINE:-16}"
OUT="${OUT:-esperimenti}"
EPOCHS="${EPOCHS:-}"            # empty = num_epochs of each script's CONFIG
EPOCH_ARGS=()
[ -n "$EPOCHS" ] && EPOCH_ARGS=(--epochs "$EPOCHS")
STATS="src/landmark_stats.py"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) — $(date)"
echo "Modes: $MODES | batch heatmap=$BATCH_HEATMAP refine=$BATCH_REFINE | epochs=${EPOCHS:-CONFIG} | output: $OUT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ── 0. Unit tests (a few seconds; stop here if something is broken) ──────────
python Test/test_unita.py

mkdir -p "$OUT"
# A pre-existing landmark_stats.py would be used by the first Stage 1: move it aside
if [ -f "$STATS" ]; then
    BACKUP="$OUT/landmark_stats_preesistente_$(date +%Y%m%d_%H%M%S).py"
    echo "Moving the existing $STATS to $BACKUP"
    mv "$STATS" "$BACKUP"
fi

for MODE in $MODES; do
    DIR="$OUT/$MODE"
    mkdir -p "$DIR"
    echo
    echo "################################################################"
    echo "# heatmap_mode = $MODE   →   $DIR   ($(date))"
    echo "################################################################"

    # Every mode starts from the same fallback statistics, so the comparison is fair
    rm -f "$STATS"

    echo "=== [$MODE] 1/3 Stage 1 training ==="
    python esegui_esperimento.py heatmap --mode "$MODE" --dir "$DIR" \
        --batch-size "$BATCH_HEATMAP" ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"} 2>&1 | tee "$DIR/1_train_heatmap.log"

    # train_heatmap regenerates src/landmark_stats.py from this Stage 1
    if [ -f "$STATS" ]; then
        cp "$STATS" "$DIR/landmark_stats.py"
    else
        echo "WARNING: $STATS was not generated — Stage 2 will use the fallback mask"
    fi

    echo "=== [$MODE] 2/3 Stage 2 training ==="
    python esegui_esperimento.py refine --dir "$DIR" \
        --batch-size "$BATCH_REFINE" ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"} 2>&1 | tee "$DIR/2_train_refine.log"

    echo "=== [$MODE] 3/3 Evaluation + uncertainty calibration ==="
    python esegui_esperimento.py evaluate --dir "$DIR" 2>&1 | tee "$DIR/3_evaluate.log"
done

# Leave src/ clean: the statistics of each mode are in $OUT/<mode>/landmark_stats.py.
# For predict_heatmap.py copy the chosen one back to src/landmark_stats.py.
rm -f "$STATS"

echo
echo "Done — $(date)"
for MODE in $MODES; do
    echo "--- $MODE ---"
    grep -E "^Global MRE|within 2.0mm|Spearman\(sigma|coverage on held-out" "$OUT/$MODE/3_evaluate.log" || true
done
