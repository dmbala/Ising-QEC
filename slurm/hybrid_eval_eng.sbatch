#!/usr/bin/env bash
#SBATCH --job-name=ising-hybrid-eval
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/hybrid_eval_%j.out
#SBATCH --error=logs/hybrid_eval_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
SHARED_OUTPUT_DIR="${REPO_ROOT}/outputs"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qec-decoder-d5-biased}"
CONFIG_NAME="${CONFIG_NAME:-config_ising_qec_d5_biased}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/hybrid/${EXPERIMENT_NAME}__${CONFIG_NAME}}"
ONNX_WORKFLOW="${ONNX_WORKFLOW:-0}"
QUANT_FORMAT="${QUANT_FORMAT:-}"
SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES:-4096}"

mkdir -p "${REPO_ROOT}/logs" "${RESULTS_DIR}"
cd "${REPO_ROOT}"
echo "[hybrid] node=$(hostname)"
nvidia-smi -L

for cfg_file in "${REPO_ROOT}"/conf/*.yaml; do
    ln -sfn "${cfg_file}" "${ISING_DIR}/conf/$(basename "${cfg_file}")"
done

for basis in X Z; do
    log_path="${RESULTS_DIR}/ablation_${basis}.log"
    echo "[hybrid] running decoder_ablation basis=${basis} -> ${log_path}"
    singularity exec --nv \
        --bind "${ISING_DIR}:${ISING_DIR}" \
        --bind "${SHARED_OUTPUT_DIR}:${SHARED_OUTPUT_DIR}" \
        --env SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}" \
        --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
        --env CONFIG_NAME="${CONFIG_NAME}" \
        --env WORKFLOW=decoder_ablation \
        --env GPUS=1 \
        --env FRESH_START=0 \
        --env ONNX_WORKFLOW="${ONNX_WORKFLOW}" \
        --env QUANT_FORMAT="${QUANT_FORMAT}" \
        --env PREDECODER_INFERENCE_NUM_SAMPLES="${SAMPLES}" \
        --env PREDECODER_INFERENCE_MEAS_BASIS="${basis}" \
        --env PREDECODER_INFERENCE_NUM_WORKERS=0 \
        --env PREDECODER_DISABLE_SDR=1 \
        --env PREDECODER_TORCH_COMPILE=0 \
        --env PREDECODER_VERBOSE=1 \
        "$SIF" bash "${ISING_DIR}/code/scripts/cluster_train.sh" >"${log_path}" 2>&1
done

python3 "${REPO_ROOT}/bench/eval_hybrid_gate.py" \
    --x-log "${RESULTS_DIR}/ablation_X.log" \
    --z-log "${RESULTS_DIR}/ablation_Z.log" \
    --output-json "${RESULTS_DIR}/hybrid_policies.json" \
    --output-md "${RESULTS_DIR}/hybrid_policies.md"

echo "[hybrid] done"
