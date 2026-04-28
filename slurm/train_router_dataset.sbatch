#!/usr/bin/env bash
#SBATCH --job-name=router-dataset
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/router_dataset_%j.out
#SBATCH --error=logs/router_dataset_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" python "${REPO_ROOT}/bench/build_regime_dataset.py" \
        --repo-root "${REPO_ROOT}"

echo "[router-dataset] done"
