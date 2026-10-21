#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 only, on an existing Stage 1, on CINECA Leonardo:
#   train Stage 2 → analisi_stage2.py → strategie_stage2.py (no --salva)
#
# DIR must already contain modello_heatmap.pth and landmark_stats.py of the
# Stage 1 to use (copied from a previous run). modello_refine.pth is written there.
#   mkdir -p logs && sbatch -A <project> --export=ALL,DIR=esperimenti_run2/s2_uniforme run_stage2_leonardo.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ceph_stage2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

DIR="${DIR:-esperimenti_run2/s2_uniforme}"
BATCH_REFINE="${BATCH_REFINE:-8}"
EPOCHS="${EPOCHS:-}"
EPOCH_ARGS=()
[ -n "$EPOCHS" ] && EPOCH_ARGS=(--epochs "$EPOCHS")

for f in modello_heatmap.pth landmark_stats.py; do
    [ -f "$DIR/$f" ] || { echo "ERROR: $DIR/$f not found (copy the Stage 1 of the run you want to use)"; exit 1; }
done
[ -f leonardo_env.sh ] || { echo "ERROR: leonardo_env.sh not found: run  bash setup_leonardo.sh  first"; exit 1; }
source leonardo_env.sh
module purge
module load "$PY_MODULE"
source "$VENV/bin/activate"
export HF_HUB_CACHE HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "nvidia-smi: no GPU visible in this job"
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', ok); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: PyTorch does not see the GPU"; exit 1; }

echo "Stage 2 only — $DIR — $(date)"
cp src/parametri_stage2.py "$DIR/parametri_stage2.py"

echo "=== 1/3 Stage 2 training ==="
python esegui_esperimento.py refine --dir "$DIR" --batch-size "$BATCH_REFINE" \
    ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"} 2>&1 | tee "$DIR/2_train_refine.log"

echo "=== 2/3 Stage 2 analysis ==="
python analisi_stage2.py --dir "$DIR" 2>&1 | tee "$DIR/analisi_stage2.log"

echo "=== 3/3 Ways of combining Stage 1 and Stage 2 (fusion re-fitted, nothing saved) ==="
python strategie_stage2.py "$DIR/analisi_stage2.csv" 2>&1 | tee "$DIR/strategie_stage2.log"

echo "Done — $(date)"
