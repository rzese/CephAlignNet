#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tutto.slurm on CINECA Leonardo (Booster, A100 64 GB).
# Needs leonardo_env.sh, written once by setup_leonardo.sh on a login node.
#
# Submit from the repository root, with your project account:
#   mkdir -p logs && sbatch -A <project> run_leonardo.slurm
# The same options as run_tutto.slurm, e.g. the second run:
#   sbatch -A <project> --export=ALL,MODES=probabilistic,STATS_DA=esperimenti,OUT=esperimenti_run2 run_leonardo.slurm
# Quick test in the debug queue (max 30 min):
#   sbatch -A <project> --qos=boost_qos_dbg --time=00:30:00 --export=ALL,EPOCHS=1,OUT=esperimenti_prova run_leonardo.slurm
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ceph_tutto
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

if [ ! -f leonardo_env.sh ]; then
    echo "ERROR: leonardo_env.sh not found: run  bash setup_leonardo.sh  on a login node first"
    exit 1
fi
source leonardo_env.sh

module purge
module load "$PY_MODULE"
source "$VENV/bin/activate"
export HF_HUB_CACHE HF_HUB_OFFLINE=1
export CONDA_ENV=none          # the venv is already active: run_tutto.slurm must not touch conda

python -c "import sys, torch; ok = torch.cuda.is_available(); \
print('torch', torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU'); \
sys.exit(0 if ok else 1)" || { echo "ERROR: PyTorch does not see the GPU"; exit 1; }

# The pipeline itself (its #SBATCH lines are ignored when run with bash)
exec bash run_tutto.slurm
