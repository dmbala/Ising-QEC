#!/usr/bin/env bash
#SBATCH --job-name=ising-sweep-submit
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sweep_train_submit_%j.out
#SBATCH --error=logs/sweep_train_submit_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
ISING_DIR="${REPO_ROOT}/Ising-Decoding"

CONFIGS="${CONFIGS:-config_ising_qec_d5_biased,config_ising_qec_d5_bias3,config_ising_qec_d5_bias30,config_ising_qec_d5_drift,config_ising_qec_d5_hotspot}"
SAMPLES="${PREDECODER_TRAIN_SAMPLES:-262144}"
EPOCHS="${PREDECODER_TRAIN_EPOCHS:-50}"
FRESH_START="${FRESH_START:-0}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-qec-decoder-d5}"

mkdir -p "${REPO_ROOT}/logs"

for cfg_file in "${REPO_ROOT}"/conf/*.yaml; do
    ln -sfn "${cfg_file}" "${ISING_DIR}/conf/$(basename "${cfg_file}")"
done

IFS=',' read -r -a cfgs <<< "${CONFIGS}"
for cfg in "${cfgs[@]}"; do
    suffix="${cfg#config_ising_qec_d5_}"
    suffix="${suffix//_/-}"
    exp_name="${EXPERIMENT_PREFIX}-${suffix}"
    job_id="$(
        sbatch --parsable \
            --export=ALL,CONFIG_NAME="${cfg}",EXPERIMENT_NAME="${exp_name}",PREDECODER_TRAIN_SAMPLES="${SAMPLES}",PREDECODER_TRAIN_EPOCHS="${EPOCHS}",FRESH_START="${FRESH_START}" \
            "${REPO_ROOT}/slurm/train_eng.sbatch"
    )"
    echo "[sweep-train] submitted ${cfg} -> ${exp_name} as ${job_id}"
done
