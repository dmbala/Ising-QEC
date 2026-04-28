# 6-Week Robustness Plan

Canonical execution plan for the next research phase on branch `codex/ising-qec-robustness-scaffold`.

## Current baseline and motivation

This repo already demonstrates a working NVIDIA Ising deployment for `d=5` surface-code decoding under biased circuit-level noise on Cannon H200 hardware:

- `model_id=1` / Fast model with the fixed `9x9x9` receptive field
- biased `Z:X = 10:1` configuration in `conf/config_ising_qec_d5_biased.yaml`
- fine-tuned FP32 checkpoint and FP8 TensorRT engine
- end-to-end round-trip latency below `10 us`
- existing sweep and hybrid-analysis scaffolding in `bench/` and `slurm/`

The next research question is not whether the pipeline works, but how robust it is under noise shift and whether a lightweight hybrid policy can improve the latency-vs-LER tradeoff when the trained pre-decoder is evaluated off its training point.

Two important constraints carry through the whole plan:

- The public NVIDIA Ising config surface only exposes global 25-parameter circuit-level noise settings.
- The added `drift` and `hotspot` configs are therefore proxy stress tests, not true time-varying drift or site-local hardware hotspot models.

## Week-by-week schedule

### Week 1: Baseline freeze and artifact capture

Goal: freeze one canonical reference point before running any robustness study.

Tasks:

- Re-run the existing `qec-decoder-d5-biased` training and inference path if needed to ensure all artifacts are present and reproducible.
- Record one canonical baseline package:
  - FP32 LER
  - FP8 TRT LER
  - PyMatching baseline and after-predecoder latency
  - full round-trip latency
- Capture the exact commands used:

```bash
sbatch --export=ALL,PREDECODER_TRAIN_SAMPLES=262144 slurm/train_eng.sbatch
sbatch slurm/export_fp8_eng.sbatch
sbatch slurm/latency_eng.sbatch
sbatch slurm/latency_trt_eng.sbatch
sbatch slurm/roundtrip_eng.sbatch
```

Deliverables:

- updated baseline log bundle under `logs/`
- canonical metrics snapshot in `results/` or a short appendix added to `SUMMARY.md`
- one note listing exact checkpoint and engine artifact names used for all later comparisons

Acceptance criteria:

- baseline Avg LER and latency numbers are reproduced within expected noise of the current repo summary
- the chosen checkpoint and engine files are fixed for the remaining weeks

### Week 2: Noise-sweep setup and matrix runs

Goal: generate the first train-vs-test robustness matrix across nearby and shifted noise points.

Tasks:

- Train one model per config:
  - `config_ising_qec_d5_biased`
  - `config_ising_qec_d5_bias3`
  - `config_ising_qec_d5_bias30`
  - `config_ising_qec_d5_drift`
  - `config_ising_qec_d5_hotspot`
- Use the existing submission wrapper:

```bash
sbatch slurm/sweep_train_eng.sbatch
```

- Run matrix evaluation across all trained experiments and eval configs:

```bash
sbatch slurm/sweep_eval_eng.sbatch
```

- Aggregate results using the generated `summary.csv`, `summary.jsonl`, and `aggregate.md`.

Deliverables:

- one complete train-experiment vs eval-config matrix under `results/matrix/<run>/`
- aggregated CSV and Markdown summary for quick inspection

Acceptance criteria:

- each trained experiment has at least one successful inference result on every eval config
- no missing rows in the matrix for the chosen experiment/config set

### Week 3: Robustness and generalization analysis

Goal: turn the raw matrix into a clear specialization-vs-transfer story.

Tasks:

- Compare on-diagonal vs off-diagonal LER behavior for all train/test pairs.
- Quantify which shifts hurt most:
  - bias reduction (`10:1 -> 3:1`)
  - bias increase (`10:1 -> 30:1`)
  - proxy drift
  - proxy hotspot
- Identify whether any single trained model generalizes better than the rest.
- Flag whether latency after pre-decoding remains stable even when LER degrades.

Deliverables:

- one short analysis note or table ranking cross-noise robustness
- one heatmap-ready CSV derived from the matrix output
- one summary statement answering: “Does specialization transfer, and where does it fail?”

Acceptance criteria:

- a preferred “robust default” training point is selected
- at least one concrete failure mode is identified from off-diagonal results

### Week 4: Residual-weight-gated hybrid evaluation

Goal: determine whether a simple gate can recover performance on difficult residuals while keeping the fast path cheap.

Tasks:

- Run decoder ablation for the most important trained checkpoint/config pair:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=qec-decoder-d5-biased,CONFIG_NAME=config_ising_qec_d5_biased slurm/hybrid_eval_eng.sbatch
```

- Use the generated ablation logs and `bench/eval_hybrid_gate.py` outputs to inspect policies of the form:
  - use `No-op` or `Union-Find` below a residual-weight threshold
  - fall back to `Corr-PM` above that threshold
- Evaluate whether the best policy differs between X and Z basis, and whether one shared threshold is still acceptable.

Deliverables:

- `hybrid_policies.json`
- `hybrid_policies.md`
- one selected hybrid policy with a concrete threshold and fallback decoder

Acceptance criteria:

- at least one gated policy is identified that is interesting on either LER or latency
- the chosen policy is simple enough to explain in one paragraph and one table

### Week 5: Latency-vs-LER Pareto analysis

Goal: convert the baseline, matrix, and hybrid results into a single decision-ready tradeoff analysis.

Tasks:

- Combine:
  - baseline FP32 / FP8 results
  - off-diagonal matrix degradation
  - hybrid gate estimates
- Compare these operating modes:
  - PyMatching alone
  - predecoder + Corr-PM
  - gated hybrid policy
- Choose one primary threshold/policy for the paper story based on the best latency-vs-LER balance.

Deliverables:

- one Pareto-style comparison table
- one recommended operating policy for later paper figures
- one sentence-level claim for the paper draft, for example:
  - robustness degrades under shift, but a simple hybrid gate recovers much of the loss

Acceptance criteria:

- one operating point is chosen and justified
- the decision is based on measured or derived metrics already present in repo outputs

### Week 6: Paper figures, tables, and writing package

Goal: package the work into a first-pass submission bundle.

Tasks:

- Produce the first figure set:
  - train-vs-test LER heatmap
  - latency-vs-LER comparison plot
  - residual-weight / hybrid-policy summary figure
- Produce the first table set:
  - baseline metrics
  - cross-noise matrix summary
  - selected hybrid-policy results
- Draft the paper skeleton:
  - problem statement
  - setup and limitations
  - robustness results
  - hybrid-policy results
  - discussion of proxy drift/hotspot limitations

Deliverables:

- figure-ready CSVs and final tables under `results/`
- a short writing package or outline that can be turned into a workshop paper draft

Acceptance criteria:

- every major claim in the draft maps to an existing artifact in the repo
- limitations of the public noise API are stated explicitly

## Concrete artifacts and expected outputs

By the end of the 6 weeks, the repo should contain or be able to regenerate:

- trained checkpoints for the selected noise configs under `outputs/outputs/<experiment>/`
- matrix results under `results/matrix/<run>/`
- hybrid results under `results/hybrid/<experiment>__<config>/`
- aggregated tables:
  - `summary.csv`
  - `aggregate.csv`
  - `aggregate.md`
  - `hybrid_policies.json`
  - `hybrid_policies.md`
- a paper-ready bundle of figures and tables derived from those files

## Experiment matrix and acceptance criteria

Minimum matrix to run and discuss:

| Train experiment | Eval config | Why it matters |
| --- | --- | --- |
| `qec-decoder-d5-biased` | `config_ising_qec_d5_biased` | in-distribution baseline |
| `qec-decoder-d5-biased` | `config_ising_qec_d5_bias3` | weaker bias shift |
| `qec-decoder-d5-biased` | `config_ising_qec_d5_bias30` | stronger bias shift |
| `qec-decoder-d5-biased` | `config_ising_qec_d5_drift` | proxy nonstationary drift |
| `qec-decoder-d5-biased` | `config_ising_qec_d5_hotspot` | proxy structured stress |
| all trained configs | all eval configs | full specialization / transfer picture |

Success criteria for the phase:

- Baseline numbers remain reproducible.
- Matrix runs complete cleanly for the selected config set.
- A robustness story is visible in the off-diagonal results.
- A simple gated hybrid policy is identified and justified.
- The repo contains enough artifacts to support a short paper draft.

## Risks, assumptions, and what not to change

Risks:

- `PREDECODER_LER_FINAL_ONLY=1` remains necessary because per-epoch LER validation is still unstable.
- FP8 calibration may still leave a small accuracy gap versus FP32.
- Validation remains tied to upstream public datagen behavior, including the Fast model’s fixed receptive field.
- Drift and hotspot remain proxy configs unless upstream datagen is extended.

Assumptions:

- All jobs continue to run on `kempner_eng` H200 nodes.
- The current Fast-model (`model_id=1`) path stays the primary baseline.
- Existing sweep and hybrid scripts remain the control surface for the research phase.

What not to change during this plan unless there is a compelling reason:

- do not switch to the Accurate model mid-study
- do not change the branch goal from robustness/hybrid evaluation to new infrastructure work
- do not redefine drift/hotspot as true hardware-local effects in the writeup; keep calling them proxy stress configurations
- do not compare results produced from mismatched checkpoints without recording the exact experiment name and artifact path
