# Regime Router

This directory contains the reusable routing subsystem for choosing among multiple fine-tuned NVIDIA Ising pre-decoders.

## Purpose

The main repo already supports:

- one fine-tuned pre-decoder per noise regime
- matrix evaluation across train/test noise mismatch
- hybrid residual-weight analysis

`regime_router/` adds the next layer:

- extract compact proxy features that summarize the current noise regime
- train a lightweight classifier to predict the best expert
- route to a regime-specific fine-tuned model or fall back to a default expert
- score routed decisions against the train-vs-test matrix already produced elsewhere in the repo

## Directory contract

- `types.py`: shared dataclasses for regime specs, feature windows, examples, and router decisions
- `catalog.py`: load `conf/regime_catalog.yaml` and `conf/regime_router.yaml`
- `features.py`: turn 25-parameter noise configs into compact proxy feature vectors
- `dataset.py`: build a proxy routing dataset from the regime catalog
- `classifier.py`: train and serve a lightweight softmax router model
- `router.py`: apply confidence-thresholded routing with a default fallback
- `evaluate.py`: compare learned routing, oracle routing, and default single-model routing

## Execution surface

Thin entrypoints live outside this package:

- `bench/build_regime_dataset.py`
- `bench/train_regime_classifier.py`
- `bench/run_routed_eval.py`
- `bench/export_regime_catalog.py`
- `slurm/train_router_dataset.sbatch`
- `slurm/train_router_model.sbatch`
- `slurm/routed_eval_eng.sbatch`

## Important limitation

The current routing dataset is a **proxy dataset**, not a true online hardware telemetry stream. It is derived from:

- the public 25-parameter noise configs
- lightweight synthetic window perturbations

That makes it appropriate for architecture bring-up and offline routing experiments, but not yet a claim about one-shot or hardware-native online classification.
