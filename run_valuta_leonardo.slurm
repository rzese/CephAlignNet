#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation only of an already trained experiment on CINECA Leonardo:
#   evaluate_refine (Stage 1 / Stage 1+2, uncertainty coverage with the
#   validation calibration) + analisi_stage2 (per-point CSV for the analyses)
#
# INSIEME: unife (external UNIFE tracings, data/tracciati), test, test_aariz,
#          test_isbi1, test_isbi2, validation. Default: unife.
# CEPH_SPLIT: the split the model was trained with (casuale for esperimenti_leo);
#             it does not matter for unife, which is never used in training.
#   mkdir -p logs && sbatch --export=ALL,DIR=esperimenti_leo/probabilistic,INSIEME=unife,CEPH_SPLIT=casuale run_valuta_leonardo.slurm
# Output in DIR: valuta_<INSIEME>.log, analisi_stage2_<INSIEME>.csv, analisi_stage2_<INSIEME>.log
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ceph_valuta
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

DIR="${DIR:-esperimenti_leo/probabilistic}"
INSIEME="${INSIEME:-unife}"
for f in modello_heatmap.pth modello_refine.pth; do
    [ -f "$DIR/$f" ] || { echo "ERROR: $DIR/$f not found"; exit 1; }
done
[ -f leonardo_env.sh ] || { echo "ERROR: leonardo_env.sh not found: run  bash setup_leonardo.sh  first"; exit 1; }
source leonardo_env.sh
module purge
module load "$PY_MODULE"
source "$VENV/bin/activate"
export HF_HUB_CACHE HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "nvidia-smi: no GPU visible in this job"
echo "Evaluation of $DIR on $INSIEME (split ${CEPH_SPLIT:-ufficiale}) — $(date)"

python esegui_esperimento.py evaluate --dir "$DIR" --insieme "$INSIEME" 2>&1 | tee "$DIR/valuta_${INSIEME}.log"
python analisi_stage2.py --dir "$DIR" --insieme "$INSIEME" 2>&1 | tee "$DIR/analisi_stage2_${INSIEME}.log"

echo "Done — $(date)"
