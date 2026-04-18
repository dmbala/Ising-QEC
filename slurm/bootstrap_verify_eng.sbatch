#!/usr/bin/env bash
#SBATCH --job-name=ising-bootstrap
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/bootstrap_%j.out
#SBATCH --error=logs/bootstrap_%j.err

# 1-epoch verification that load_checkpoint picks up Fast pretrained weights
# when dropped at outputs/<EXP>/models/PreDecoderModelMemory_v1.0.0.pt.
# If the log shows "Loaded model state dict ..." then the fine-tune bootstrap works.

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
SHARED_OUTPUT_DIR="${REPO_ROOT}/outputs"

EXPERIMENT_NAME=qec-decoder-d5-biased
CONFIG_NAME=config_ising_qec_d5_biased
FAST_PT="${ISING_DIR}/models/Ising-Decoder-SurfaceCode-1-Fast.pt"

MODEL_DIR="${SHARED_OUTPUT_DIR}/outputs/${EXPERIMENT_NAME}/models"
BOOTSTRAP_PT="${MODEL_DIR}/PreDecoderModelMemory_v1.0.0.pt"

mkdir -p "${MODEL_DIR}"
cd "$REPO_ROOT"

if [ ! -f "$BOOTSTRAP_PT" ]; then
    echo "[bootstrap] seeding ${BOOTSTRAP_PT} from Fast pretrained weights"
    cp "$FAST_PT" "$BOOTSTRAP_PT"
else
    echo "[bootstrap] ${BOOTSTRAP_PT} already present, leaving alone"
fi

echo "[bootstrap] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"

singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --bind "${SHARED_OUTPUT_DIR}:${SHARED_OUTPUT_DIR}" \
    --env SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}" \
    --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    --env CONFIG_NAME="${CONFIG_NAME}" \
    --env WORKFLOW=train \
    --env GPUS=1 \
    --env FRESH_START=0 \
    --env PREDECODER_DISABLE_SDR=1 \
    --env PREDECODER_TORCH_COMPILE=0 \
    --env PREDECODER_VERBOSE=1 \
    --env PREDECODER_TRAIN_EPOCHS=1 \
    --env PREDECODER_TRAIN_SAMPLES=4096 \
    --env PREDECODER_VAL_SAMPLES=512 \
    --env PREDECODER_TEST_SAMPLES=512 \
    "$SIF" bash "${ISING_DIR}/code/scripts/cluster_train.sh"

echo "[bootstrap] done"
