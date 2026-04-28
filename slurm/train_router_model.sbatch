#!/usr/bin/env bash
#SBATCH --job-name=router-train
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/router_train_%j.out
#SBATCH --error=logs/router_train_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/results/router/datasets/regime_router_dataset.csv}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" python "${REPO_ROOT}/bench/train_regime_classifier.py" \
        --repo-root "${REPO_ROOT}" \
        --dataset "${DATASET_PATH}"

echo "[router-train] done"
