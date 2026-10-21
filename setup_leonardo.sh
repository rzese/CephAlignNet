#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup of the Python environment on CINECA Leonardo.
# Run it on a LOGIN node (they have internet, compute nodes do not), from the
# repository root:
#     bash setup_leonardo.sh
# Options (environment variables):
#     BASE       where venv and model cache go           (default $WORK/$USER)
#     PY_MODULE  Python module to load                   (default: python)
#
# It creates:
#   $BASE/venv_ceph      Python venv with requirements.txt
#   $BASE/hf_cache       pretrained HRNet-W48 weights (read offline by the jobs)
#   leonardo_env.sh      the paths above, read by run_leonardo.slurm
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${WORK:-}" ] && [ -z "${BASE:-}" ]; then
    echo "ERROR: \$WORK is not set. Set BASE to a folder in your project area, e.g. BASE=/leonardo_work/<project>/$USER"
    exit 1
fi
BASE="${BASE:-$WORK/$USER}"
VENV="$BASE/venv_ceph"
HF_HUB_CACHE="$BASE/hf_cache"
PY_MODULE="${PY_MODULE:-python}"
mkdir -p "$BASE" "$HF_HUB_CACHE"

# ── Python ───────────────────────────────────────────────────────────────────
module purge
if ! module load "$PY_MODULE"; then
    echo "ERROR: cannot load module '$PY_MODULE'. See the available ones with:  module avail python"
    echo "then re-run with e.g.  PY_MODULE=python/3.11.6--gcc--8.5.0 bash setup_leonardo.sh"
    exit 1
fi
echo "Python: $(python3 --version) ($(which python3))"

# ── venv + packages ──────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# ── pretrained weights into the shared cache (compute nodes are offline) ─────
export HF_HUB_CACHE
python - <<'EOF'
import timm
timm.create_model("hrnet_w48", pretrained=True, features_only=True, in_chans=1)
print("HRNet-W48 weights downloaded")
EOF
if ! ls "$HF_HUB_CACHE"/models--timm--hrnet_w48.ms_in1k/snapshots/*/model.safetensors >/dev/null 2>&1; then
    echo "ERROR: weights not found in $HF_HUB_CACHE"
    exit 1
fi

# ── paths for the jobs ───────────────────────────────────────────────────────
cat > leonardo_env.sh <<EOF
# Written by setup_leonardo.sh — read by run_leonardo.slurm
PY_MODULE="$PY_MODULE"
VENV="$VENV"
HF_HUB_CACHE="$HF_HUB_CACHE"
EOF
echo "Wrote leonardo_env.sh"

# ── checks ───────────────────────────────────────────────────────────────────
python Test/test_unita.py
for d in data/Aariz data/Cepha400; do
    [ -d "$d" ] || echo "WARNING: $d not found — copy the datasets into data/ before launching"
done
echo
echo "Done. Quick test on a GPU (30 min debug queue):"
echo "  mkdir -p logs && sbatch -A <project> --qos=boost_qos_dbg --time=00:30:00 --export=ALL,EPOCHS=1,OUT=esperimenti_prova run_leonardo.slurm"
