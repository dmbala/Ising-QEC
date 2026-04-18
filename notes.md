Build the needed packages using the Singularity container apporach. 
https://github.com/NVIDIA/ising

---

# Project: Real-Time QEC Decoding via NVIDIA Ising

## 1. Context & Objective
As of April 2026, **NVIDIA Ising** is the state-of-the-art for hybrid quantum-classical control. This project implements a **Hardware-Aware Virtual Node** to perform real-time error decoding using 3D CNNs on H100/H200 hardware, aiming for sub-$10\mu s$ latency.

## 2. Technical Stack
* **Quantum OS:** `CUDA-Q` (Target: `stim` for high-speed syndrome generation).
* **Core Model:** `Ising-Decoder-SurfaceCode-1-Accurate` (1.79M parameters).
* **Precision:** `FP8` (E4M3 format) using `NVIDIA ModelOpt`.
* **Inference:** `TensorRT` + `CUDA Graphs`.
* **Hardware:** NVIDIA H100/H200 (Hopper/Blackwell architecture).

---

## 3. Project Plan (Steps for Agents)

### Phase 1: Environment & Virtual Lattice
* [ ] **Initialize `CUDA-Q` Container:** Use `nvcr.io/nvidia/cuda-quantum:latest`.
* [ ] **Define Code Topology:** Set up a Distance $d=5$ Surface Code lattice.
* [ ] **Inject Refined Noise:** Implement an asymmetric noise model (e.g., $10:1$ $Z/X$ bias) and localized crosstalk "hotspots."

### Phase 2: Diagnostic & Data Generation
* [ ] **Generate Syndrome Volumes:** Simulate 100k+ shots to create a $13 \times 13 \times 13$ tensor (Space $X$, Space $Y$, Time $T$).
* [ ] **Heatmap Verification:** Visualize stabilizer trigger frequencies to identify hardware "weak spots."
* [ ] **Export Dataset:** Format data as `.pt` (PyTorch) for the Ising training framework.

### Phase 3: Hardware-Aware Fine-Tuning
* [ ] **Load Base Weights:** Pull `ising-decoder-sc1-accurate` weights.
* [ ] **Transfer Learning:** Fine-tune for 50 epochs on the custom "hotspot" dataset using the Ising training framework.
* [ ] **Validation:** Verify the decoding threshold against a standard `PyMatching` baseline.

### Phase 4: Precision Engineering (Quantization)
* [ ] **Calibration:** Use `ModelOpt` to analyze the dynamic range of syndromes.
* [ ] **FP8 Conversion:** Insert Q/DQ nodes and export to ONNX.
* [ ] **Engine Compilation:** Build a `TensorRT` engine with `--useCudaGraph` and `--fp8` flags.

### Phase 5: Real-Time Benchmark
* [ ] **Latency Test:** Measure end-to-end "Round-Trip" time (Simulator $\rightarrow$ GPU $\rightarrow$ Correction).
* [ ] **Final Metric:** Target $<10\mu s$ latency with $>1.5\text{x}$ accuracy gain over algorithmic decoders.

---

## 4. Key 2026 Constraints
> **Note:** When deploying the Ising-Decoder, ensure the `receptive field` matches the training volume. The **Accurate** model is trained on $13 \times 13 \times 13$, while the **Fast** model is trained on $9 \times 9 \times 9$. For hardware-agnostic research, $d=5$ (Accurate) is the recommended balance for H100 benchmarks.



