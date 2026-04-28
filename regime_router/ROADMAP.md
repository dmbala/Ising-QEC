# Regime Router Roadmap

## Objective

Build a regime-aware decoder-selection layer that chooses among `N` fine-tuned NVIDIA Ising experts using slow-timescale noise classification.

## Target architecture

1. A catalog defines the supported regimes and their associated training configs and artifact paths.
2. A feature extractor converts calibration-like metadata and rolling syndrome summaries into a compact feature vector.
3. A lightweight classifier predicts the most likely regime and a confidence score.
4. A router either selects the predicted expert or falls back to the default regime if confidence is low.
5. The selected expert acts as the pre-decoder before `Corr-PM` / PyMatching.

## Milestones

### M1: Scaffolding

- add `regime_router/` package
- add router configs in `conf/`
- add dataset, training, and routed-eval CLI entrypoints

### M2: Offline routing experiments

- generate a proxy routing dataset from the existing five regimes
- train a first-pass classifier
- evaluate routed performance using the matrix outputs already produced by `slurm/sweep_eval_eng.sbatch`

### M3: Quantization-aware routing

- compare router behavior against both torch and FP8 matrix summaries
- measure whether routed FP8 degrades more than routed FP32 under shift

### M4: Hardware-informed routing

- replace or augment proxy features with real calibration summaries
- optionally incorporate rolling syndrome-window statistics from hardware or simulator logs

## Success conditions

- the routed policy beats the single default expert on average shifted-noise LER
- the learned router approaches oracle routing closely enough to justify deployment
- the subsystem stays interpretable enough to explain in a paper and to debug in production
