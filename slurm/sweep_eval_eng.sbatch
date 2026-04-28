#!/usr/bin/env bash
#SBATCH --job-name=ising-eval-matrix
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_matrix_%j.out
#SBATCH --error=logs/eval_matrix_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/matrix/${SLURM_JOB_ID}}"

TRAIN_EXPERIMENTS="${TRAIN_EXPERIMENTS:-qec-decoder-d5-biased,qec-decoder-d5-bias3,qec-decoder-d5-bias30,qec-decoder-d5-drift,qec-decoder-d5-hotspot}"
EVAL_CONFIGS="${EVAL_CONFIGS:-config_ising_qec_d5_biased,config_ising_qec_d5_bias3,config_ising_qec_d5_bias30,config_ising_qec_d5_drift,config_ising_qec_d5_hotspot}"
EVAL_MODES="${EVAL_MODES:-torch,fp8}"

mkdir -p "${REPO_ROOT}/logs" "${RESULTS_DIR}"
cd "${REPO_ROOT}"
echo "[eval-matrix] node=$(hostname)"
nvidia-smi -L

torch_csv=""
fp8_csv=""
IFS=',' read -r -a modes <<< "${EVAL_MODES}"
for mode in "${modes[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    case "${mode}" in
        torch)
            mode_dir="${RESULTS_DIR}/torch"
            onnx_workflow="0"
            quant_format=""
            mode_label="torch"
            ;;
        fp8)
            mode_dir="${RESULTS_DIR}/fp8"
            onnx_workflow="2"
            quant_format="fp8"
            mode_label="fp8"
            ;;
        *)
            echo "[eval-matrix] unknown mode '${mode}' (expected torch or fp8)" >&2
            exit 1
            ;;
    esac

    python3 "${REPO_ROOT}/bench/eval_ler_matrix.py" \
        --repo-root "${REPO_ROOT}" \
        --ising-dir "${ISING_DIR}" \
        --sif "${SIF}" \
        --shared-output-dir "${REPO_ROOT}/outputs" \
        --results-root "${mode_dir}" \
        --train-experiments "${TRAIN_EXPERIMENTS}" \
        --eval-configs "${EVAL_CONFIGS}" \
        --samples "${PREDECODER_INFERENCE_NUM_SAMPLES:-4096}" \
        --latency-samples "${PREDECODER_INFERENCE_LATENCY_SAMPLES:-0}" \
        --onnx-workflow "${onnx_workflow}" \
        --quant-format "${quant_format}" \
        --mode-label "${mode_label}"

    python3 "${REPO_ROOT}/bench/aggregate_results.py" "${mode_dir}/json"

    if [ "${mode}" = "torch" ]; then
        torch_csv="${mode_dir}/json/aggregate.csv"
    fi
    if [ "${mode}" = "fp8" ]; then
        fp8_csv="${mode_dir}/json/aggregate.csv"
    fi
done

if [ -n "${torch_csv}" ] && [ -n "${fp8_csv}" ]; then
    python3 "${REPO_ROOT}/bench/compare_quant_modes.py" \
        --torch-csv "${torch_csv}" \
        --fp8-csv "${fp8_csv}" \
        --csv-out "${RESULTS_DIR}/quantization_gap.csv" \
        --md-out "${RESULTS_DIR}/quantization_gap.md"
fi

echo "[eval-matrix] done"
