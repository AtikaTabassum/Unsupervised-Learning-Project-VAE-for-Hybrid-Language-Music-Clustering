# Project: Audio + Lyrics Clustering (425-project)

This repository contains code and notebooks for extracting audio/lyrics features, training VAE/AEs, and comparing clustering methods across three task recipes (easy, medium, hard).

---

## Quick start

1. Create a Python environment (recommended Python 3.9+ / 3.11):

   pip install -r requirements.txt

   If `requirements.txt` is not available, install core packages:

   pip install torch torchvision torchaudio librosa scikit-learn matplotlib sentence-transformers tqdm pandas seaborn

2. Quick smoke runs (small SAMPLE_SIZE recommended for fast checks):

   - Easy: SAMPLE_SIZE=20 EPOCHS=1 python run_easy_task.py
   - Medium: SAMPLE_SIZE=20 EPOCHS=1 python run_medium_task.py
   - Hard (resolve-only path check): RESOLVE_PATHS_ONLY=1 python run_hard_task.py

Environment variables supported:
- SAMPLE_SIZE (limit dataset for quick tests)
- EPOCHS (override training epochs)
- BATCH_SIZE, LR, BETA (training hyperparameters)
- N_CLUSTERS (for medium task when no labels are used)
- RESOLVE_PATHS_ONLY (1 = only resolve file paths without feature extraction)

---

## Top-level layout

- `run_easy_task.py` — easy recipe (FC-VAE on mel-spectrograms): trains VAE, runs KMeans on latent, PCA+KMeans baseline, t‑SNE visualization. Outputs to `results/`.
- `run_medium_task.py` — medium recipe (ConvVAE multimodal): trains conv VAE on audio+lyrics, runs KMeans / Agglomerative / DBSCAN on latent, includes **PCA (hybrid)** baseline (flattened audio + lyrics). Uses t‑SNE for visualization and writes consolidated `clusters_medium.csv`.
- `run_hard_task.py` — hard recipe (conditional/Beta-VAE): trains Beta-VAE with optional genre conditioning, saves reconstructions, computes multiple baselines (AE, PCA, spectral), and writes indexes/metrics.

- `exploratory.ipynb` — general exploratory notebook (EDA and preprocessing templates).
- `notebooks/easy.ipynb`, `notebooks/medium.ipynb`, `notebooks/hard.ipynb` — task-specific notebooks that reproduce the key steps from the corresponding `run_*` scripts and show smoke-run examples and result previews.

- `data/` — dataset folder
  - `metadata.csv` — dataset metadata (audio_path, lyrics_path, optional columns such as genre, language, etc.)
  - `audio/` and `lyrics/` subfolders with the raw files

- `results/` — outputs from experiments
  - `clustering_metrics_{easy,medium,hard}.csv` — full metric tables per method
  - `clustering_indices_{easy,medium,hard}.csv` — compact index summaries (per-user spec)
  - `latent_visualization/{easy,medium,hard}/` — t‑SNE images and comparison figures
  - `reconstructions/hard/` — saved VAE reconstructions for hard
  - `clusters_medium.csv`, `clusters_hard.csv` — consolidated cluster assignments (original metadata columns + cluster_* columns)

- `src/` — source modules
  - `dataset_easy.py`, `dataset_medium.py` — feature extraction helpers (audio mel-spectrograms, lyrics embeddings)
  - `vae_easy.py`, `convae_medium.py`, `convae_hard.py`, `MultiModalBetaVAE` — model implementations
  - `clustering_easy.py`, `clustering_medium.py`, `clustering_hard.py` — clustering wrappers and helpers (KMeans, Agglomerative, DBSCAN, PCA+KMeans)
  - `evaluation_*.py` — per-task evaluation metric calculations
  - `visualization_hard.py` — latent visualizations and reconstruction helpers

- `scripts/` — utility scripts (file checks, cluster consolidation)
- `tests/` — basic integration / unit tests (e.g., `test_integration.py`)

---

## Design notes & conventions

- t‑SNE is used as the default 2D projection for visuals; perplexity adapts to sample size automatically.
- Medium task: **no** genre-based distribution plots and no use of genre labels for clustering by default — set `N_CLUSTERS` to control k when labels are absent.
- Easy task: primary comparison is VAE latent + KMeans and PCA + KMeans baseline; metrics for Easy include Silhouette and Calinski–Harabasz indices.
- Hard task: includes reconstructions, per-language/genre distributions (if metadata present), and comparisons among VAE, AE, PCA, and spectral baselines.

---

## Reproducing & extending experiments

- To reproduce results, start with a small SAMPLE_SIZE for debugging, run each `run_*_task.py`, inspect CSVs under `results/`, and iterate hyperparameters via env vars.
- To add a new baseline or metric: add the computation in the corresponding `run_*_task.py` and append to the `metrics` list — the code writes metrics CSVs and indices automatically.

---

If you'd like, I can:
- Add annotated metric values on the comparison images for easier visual comparison ✅
- Add unit tests ensuring index CSVs and comparison images are generated after a smoke run ✅

If you want the README wording adjusted or more detail for any file, tell me which parts to expand.