# pytorch-boltz-surrogate-benchmark

A sample-efficient virtual screening pipeline benchmarking **PyTorch MC-Dropout surrogate models** against **`scikit-learn` baselines** (Random Forest, Gaussian Process, Bagged SVR, Tanimoto kNN) using **local open-source Boltz-2** as a structural binding affinity oracle.

> **Status:** Active development — offline dataset ingestion, RDKit Morgan featurisation, unit tests, and `scikit-learn` uncertainty baselines are implemented.

---
## Overview & Motivation

Modern biomolecular foundation models (such as open-source **Boltz-2**) accurately predict protein-ligand 3D structures and binding affinities, but running full 3D co-folding inference across thousands of candidate molecules on local consumer hardware is computationally expensive.

This project implements a **100% local Active-Learning Virtual Screening Loop** designed for High-Dimensional, Low-Sample-Size (HDLSS) regimes:
1. **Featurisation:** Convert small-molecule SMILES strings into 2048-bit Morgan fingerprints (`RDKit` → `NumPy`).
2. **Seed Oracle Query:** Evaluate an initial random batch against a target protein FASTA sequence using local Boltz-2 inference (with SQLite/JSON caching so no molecule is ever folded twice).
3. **Surrogate Training & Uncertainty Estimation:** Train a lightweight surrogate model on the labelled fingerprints and estimate predictive uncertainty (σ) across the unlabelled pool.
4. **Acquisition & Iteration:** Select the most promising candidates using Upper Confidence Bound (μ + βσ), run local Boltz-2 on the selected batch, and retrain iteratively.

---

## Current Repository Structure

```text
├── configs/               # Experiment configurations (in progress)
├── data/
│   ├── raw/               # Raw SMILES libraries (e.g., MoleculeNet BACE)
│   ├── processed/         # Cached RDKit Morgan fingerprint .npy arrays
│   └── oracle_cache/      # Local cache for Boltz-2 structural & affinity predictions
├── notebooks/             # Exploratory notebooks
├── results/               # Saved benchmark figures and metrics
├── src/
│   ├── __init__.py
│   ├── data_download.py   # Downloads the offline MoleculeNet BACE benchmark dataset
│   ├── features.py        # RDKit SMILES-to-Morgan fingerprint NumPy pipeline
│   └── baselines.py       # Scikit-learn RF, GP, Bagged SVR, and Tanimoto kNN surrogates
└── tests/
│   ├── __init__.py
│   └── test_features.py   # Unit tests for MorganFeaturizer and SMILES edge cases
```

---

## Getting Started

### 1. Environment Setup
(I recommend setting up a custom Conda environment).
Clone the repository and set up the Conda environment:

```bash
git clone https://github.com/JennieCouchman/pytorch-boltz-surrogate-benchmark.git
cd pytorch-boltz-surrogate-benchmark

conda create -n boltz-project python=3.11 -y
conda activate boltz-project
conda install -c conda-forge numpy pandas scipy scikit-learn rdkit matplotlib seaborn pyyaml python-dotenv ipykernel jupyterlab nomkl -y
pip install torch boltz-api
```

### 2. Download Data & Extract Features
Download the offline BACE benchmark dataset and generate cached Morgan fingerprint arrays:

```bash
python src/data_download.py
python src/features.py
```

### 3. Run Unit Tests
Verify the feature extraction pipeline:

```bash
python -m unittest discover -s tests -v
```
---

## Roadmap
- [x] Repository scaffold and offline BACE dataset downloader
- [x] RDKit Morgan fingerprint pipeline (`src/features.py`) and unit tests
- [ ] `scikit-learn` baseline suite (`src/baselines.py`)
