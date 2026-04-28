#!/usr/bin/env bash
#SBATCH --job-name=router-eval
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/router_eval_%j.out
#SBATCH --error=logs/router_eval_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/results/router/datasets/regime_router_dataset.csv}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/results/router/models/regime_router_softmax.json}"
MATRIX_CSV="${MATRIX_CSV:-}"

if [ -z "${MATRIX_CSV}" ]; then
    echo "[router-eval] MATRIX_CSV must point to an aggregate matrix CSV, e.g. results/matrix/<run>/fp8/json/aggregate.csv" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" python "${REPO_ROOT}/bench/run_routed_eval.py" \
        --repo-root "${REPO_ROOT}" \
        --dataset "${DATASET_PATH}" \
        --model "${MODEL_PATH}" \
        --matrix-csv "${MATRIX_CSV}"

echo "[router-eval] done"
