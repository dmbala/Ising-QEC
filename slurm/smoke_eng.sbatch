#!/usr/bin/env bash
#SBATCH --job-name=ising-smoke
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
SHARED_OUTPUT_DIR="${REPO_ROOT}/outputs"

mkdir -p "$SHARED_OUTPUT_DIR"
cd "$REPO_ROOT"

echo "[smoke] node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

# 1. Import sanity check.
singularity exec --nv "$SIF" python - <<'PY'
import torch, stim, pymatching
try:
    import tensorrt
    trt_v = tensorrt.__version__
except Exception as e:
    trt_v = f"unavailable ({e})"
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("stim", stim.__version__, "pymatching", pymatching.__version__, "tensorrt", trt_v)
PY

# 2. Upstream short train + inference smoke run.
#    Uses code/scripts/smoke_run.sh which sets timing-mode env vars for a 1-epoch,
#    small-sample pass. Hydra writes under $REPO_ROOT/Ising-Decoding/outputs/ by default.
singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --bind "${SHARED_OUTPUT_DIR}:${SHARED_OUTPUT_DIR}" \
    --env SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}" \
    --env PREDECODER_TORCH_COMPILE=0 \
    --env PREDECODER_DISABLE_SDR=1 \
    "$SIF" bash "${ISING_DIR}/code/scripts/smoke_run.sh"

echo "[smoke] done"
