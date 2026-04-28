# Regime Router Execution Plan

This execution plan is the subsystem-local companion to the top-level [6-week roadmap](../PLAN_6WEEK_ROBUSTNESS.md).

## Phase 1: Catalog and dataset bring-up

1. Populate `conf/regime_catalog.yaml` with the five proxy regimes:
   - `bias3`
   - `bias10`
   - `bias30`
   - `drift`
   - `hotspot`
2. Generate a proxy dataset:

```bash
sbatch slurm/train_router_dataset.sbatch
```

3. Inspect the exported catalog and dataset metadata:

```bash
python3 bench/export_regime_catalog.py
```

## Phase 2: Train the router

1. Train the first softmax classifier:

```bash
sbatch slurm/train_router_model.sbatch
```

2. Review:
   - train/valid accuracy
   - confusion matrix
   - most commonly confused regimes

## Phase 3: Evaluate routed decoding

1. Produce matrix outputs first:

```bash
sbatch slurm/sweep_eval_eng.sbatch
```

2. Evaluate routed performance against one matrix mode:

```bash
sbatch --export=ALL,MATRIX_CSV=/path/to/aggregate.csv slurm/routed_eval_eng.sbatch
```

3. Compare:
   - learned routing
   - oracle routing
   - default regime routing

## Phase 4: Quantization-aware analysis

Repeat routed evaluation for both:

- `results/matrix/<run>/torch/json/aggregate.csv`
- `results/matrix/<run>/fp8/json/aggregate.csv`

Then compare whether routing remains beneficial once the pre-decoder path is quantized.

## Deliverables

- routing dataset under `results/router/datasets/`
- trained classifier under `results/router/models/`
- routed evaluation report under `results/router/evals/`
- artifact manifest from `bench/export_regime_catalog.py`

## Defaults

- default regime: `bias10`
- confidence threshold: `0.70`
- dataset source: config-derived proxy windows
- routed evaluation split: `test`
