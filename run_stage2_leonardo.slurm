#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 only, on an existing Stage 1, on CINECA Leonardo:
#   train Stage 2 (variant VARIANTE) → analysis on validation → fusion parameters
#   → evaluation on validation, official test sets and UNIFE
#
# The Stage 1 (modello_heatmap.pth, landmark_stats.py) is taken from STAGE1_DA
# (linked / copied into DIR); modello_refine.pth and all the results go to DIR.
#   mkdir -p logs && sbatch --export=ALL,VARIANTE=B,DIR=esperimenti_s2/B run_stage2_leonardo.slurm
#   sbatch --export=ALL,VARIANTE=A,DIR=esperimenti_s2/A run_stage2_leonardo.slurm
# Variants: see VARIANTI in train_refine.py ("base" = the current Stage 2).
#
# Fusion: for VARIANTE=base the fixed parameters of src/parametri_stage2.py;
# for a new variant sigma_S2 has a different scale, so c and p are re-fitted
# on the VALIDATION set (strategie_stage2.py --salva DIR/parametri_stage2.py,
# 2-fold CV on even/odd images) and then used unchanged on test sets and UNIFE.
# Needs the resnet34 ImageNet weights in the offline cache (see setup_leonardo.sh).
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
#SBATCH --account=IscrC_NESY

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

VARIANTE="${VARIANTE:-B}"
DIR="${DIR:-esperimenti_s2/$VARIANTE}"
STAGE1_DA="${STAGE1_DA:-esperimenti_crop/probabilistic}"
BATCH_REFINE="${BATCH_REFINE:-16}"
EPOCHS="${EPOCHS:-}"
EPOCH_ARGS=()
[ -n "$EPOCHS" ] && EPOCH_ARGS=(--epochs "$EPOCHS")

for f in modello_heatmap.pth landmark_stats.py; do
    [ -f "$STAGE1_DA/$f" ] || { echo "ERROR: $STAGE1_DA/$f not found"; exit 1; }
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
if [ "$VARIANTE" != "base" ] && ! ls "$HF_HUB_CACHE"/models--timm--resnet34*/snapshots/*/model.safetensors >/dev/null 2>&1; then
    echo "ERROR: resnet34 weights not in $HF_HUB_CACHE. On a login node run:"
    echo "  source leonardo_env.sh && source \$VENV/bin/activate && HF_HUB_CACHE=\$HF_HUB_CACHE python -c \"import timm; timm.create_model('resnet34', pretrained=True)\""
    exit 1
fi

mkdir -p "$DIR"
echo "Stage 2 variant $VARIANTE — Stage 1 from $STAGE1_DA — output $DIR — $(date)"
ln -sfn "$(cd "$STAGE1_DA" && pwd)/modello_heatmap.pth" "$DIR/modello_heatmap.pth"
cp "$STAGE1_DA/landmark_stats.py" "$DIR/landmark_stats.py"

echo "=== 1/5 Stage 2 training ==="
python esegui_esperimento.py refine --dir "$DIR" --variante "$VARIANTE" --batch-size "$BATCH_REFINE" \
    ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"} 2>&1 | tee "$DIR/2_train_refine.log"

echo "=== 2/5 Stage 2 analysis on validation ==="
python analisi_stage2.py --dir "$DIR" 2>&1 | tee "$DIR/4_analisi_stage2.log"

echo "=== 3/5 Fusion parameters ==="
if [ "$VARIANTE" = "base" ]; then
    cp src/parametri_stage2.py "$DIR/parametri_stage2.py"
    python strategie_stage2.py "$DIR/analisi_stage2.csv" 2>&1 | tee "$DIR/5_strategie_stage2.log"
else
    python strategie_stage2.py "$DIR/analisi_stage2.csv" --salva "$DIR/parametri_stage2.py" 2>&1 \
        | tee "$DIR/5_strategie_stage2.log"
fi

echo "=== 4/5 Evaluation on validation (+ uncertainty calibration) ==="
python esegui_esperimento.py evaluate --dir "$DIR" 2>&1 | tee "$DIR/3_evaluate.log"

echo "=== 5/5 Official test sets and UNIFE ==="
INSIEMI="test_aariz test_isbi1 test_isbi2"
[ -d data/tracciati ] && INSIEMI="$INSIEMI unife"
for T in $INSIEMI; do
    python esegui_esperimento.py evaluate --dir "$DIR" --insieme "$T" 2>&1 | tee "$DIR/6_${T}.log"
    python analisi_stage2.py --dir "$DIR" --insieme "$T" 2>&1 | tee "$DIR/7_analisi_${T}.log"
done

echo
echo "Done — $(date)"
for f in 3_evaluate $(for T in $INSIEMI; do echo "6_$T"; done); do
    echo "  [$f]"; grep -E "^Global MRE|within 2.0mm" "$DIR/$f.log" | tail -2 || true
done
