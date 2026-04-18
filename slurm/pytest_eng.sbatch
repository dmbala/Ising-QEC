#!/usr/bin/env bash
#SBATCH --job-name=ising-pytest
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pytest_%j.out
#SBATCH --error=logs/pytest_%j.err

# Run upstream test suite inside the Singularity container to confirm our env matches CI.

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"

cd "$REPO_ROOT"
echo "[pytest] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"

# Install pytest reqs (not in train-cu12.txt), then run the test tiers.
singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --env PYTHONPATH="${ISING_DIR}/code" \
    "$SIF" bash -c "
        set -euo pipefail
        pip install --user --no-cache-dir -r '${ISING_DIR}/code/requirements_ci.txt'
        cd '${ISING_DIR}'
        python -m pytest code/tests -x -q 2>&1 | tail -200
    "

echo "[pytest] done"
