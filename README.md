# Music Clustering with Variational Autoencoders (425-project)

This repository implements an unsupervised pipeline to cluster hybrid-language music tracks using Variational Autoencoders (VAEs). It extracts audio and lyrics features, trains models at three complexity levels (easy / medium / hard), evaluates clustering quality against multiple baselines, and produces visuals and reconstruction artifacts.

---

## Project overview

Three task recipes are provided:

- **Easy Task** — VAE on mel-spectrograms (compact experiment for fast iteration). Primary baseline: PCA + KMeans.
- **Medium Task** — ConvVAE on hybrid audio + lyrics features (multimodal). Baselines: PCA (hybrid) and clustering algorithms (KMeans, Agglomerative, DBSCAN).
- **Hard Task** — Beta / conditional VAE with reconstructions and more extensive baselines (AE, PCA, spectral).

Each task produces clustering metrics, compact indices, t‑SNE visualizations, and (for hard) reconstruction images.

---

## Repository structure

```
425-project/
├── data/                      # Dataset (metadata.csv, audio/, lyrics/)
├── notebooks/                 # Exploratory notebooks (exploratory.ipynb
├── results/                   # Outputs (metrics CSVs, visualizations, reconstructions)
│   ├── clustering_metrics_*.csv
│   ├── clustering_indices_*.csv
│   ├── latent_visualization/
│   │   ├── easy/
│   │   ├── medium/
│   │   └── hard/
│   └── reconstructions/hard/
├── run_easy_task.py           # Easy recipe (VAE on mel-spectrograms)
├── run_medium_task.py         # Medium recipe (ConvVAE multimodal, PCA_hybrid baseline)
├── run_hard_task.py           # Hard recipe (Beta/conditional VAE, reconstructions)
├── src/                       # Source modules (datasets, models, clustering, evaluation)
└── README.md                  # This file
└── requirements.txt                
```

---

## Installation & Requirements

1. Create and activate a Python virtual environment (recommended Python 3.9+ / 3.11):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install pinned dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

3. PyTorch (GPU builds):

- If you plan to use a GPU, install a PyTorch wheel matching your CUDA version. Visit https://pytorch.org/get-started/locally and choose the correct command. Example for CUDA 12.2:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu122 torch==2.2.0 torchvision==0.15.2 torchaudio==2.2.2
```

4. Verify installation (quick check):

```python
python -c "import torch, librosa, pandas, sklearn; print('OK', torch.__version__)"
```

---

## Requirements file

A pinned `requirements.txt` is included at the repository root (recommended for reproducibility). Key packages include:

- `torch`, `torchvision`, `torchaudio` (PyTorch 2.x)
- `librosa`, `soundfile`
- `sentence-transformers` (lyrics embeddings)
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

You can find the full pinned versions in `requirements.txt`.

---

## Data layout & quick checks

- Required metadata file: `data/metadata.csv` (rows should include audio and lyrics paths or identifiers).
- Audio files: `data/audio/` (may contain subfolders, e.g., `00_mp3/`)
- Lyrics files: `data/lyrics/` (optional; used by medium/hard tasks)

Quick path sanity check (fast, no heavy feature extraction):

```powershell
$env:RESOLVE_PATHS_ONLY=1; python run_hard_task.py
```

Or in the notebook, use the loader in `src.dataset_medium.load_multimodal_features(resolve_paths_only=True)` to validate paths.

---

## Running the tasks (examples)

Run with PowerShell environment variables (examples):

- Easy (quick smoke):
```powershell
$env:SAMPLE_SIZE=20; $env:EPOCHS=1; python run_easy_task.py
```

- Medium (quick smoke):
```powershell
$env:SAMPLE_SIZE=20; $env:EPOCHS=1; python run_medium_task.py
```

- Hard (path check):
```powershell
$env:RESOLVE_PATHS_ONLY=1; python run_hard_task.py
```

For full experiments remove `SAMPLE_SIZE` and increase `EPOCHS` (the hard task default `EPOCHS` is set to 100 for stability).

---

## Outputs & naming conventions

Primary output locations (produced by `run_*_task.py` scripts):

- `results/clustering_metrics_{easy,medium,hard}.csv` — full per-method metrics (rows = methods/configs)
- `results/clustering_indices_{easy,medium,hard}.csv` — compact index summaries per user specification
- `results/latent_visualization/{easy,medium,hard}/` — t‑SNE images and per-method comparison figures
- `results/reconstructions/hard/` — saved reconstruction images for the hard task
- `results/clusters_medium.csv`, `results/clusters_hard.csv` — consolidated per-sample cluster labels appended to metadata

---

## Metrics computed (by task)

- **Easy**: Silhouette, Calinski–Harabasz
- **Medium**: Silhouette, Davies–Bouldin, ARI (if labels available)
- **Hard**: Silhouette, NMI, ARI, Purity

All metrics are saved into the `results/` CSV files listed above.

---

## Visualizations & how to preview them (code snippet)

The `notebooks/exploratory.ipynb` includes cells that automatically load metrics and preview images. Example code used in the notebook:

```python
import glob, os, pandas as pd
# load metrics
metrics = pd.read_csv('results/clustering_metrics_medium.csv')
# list visuals
viz_files = sorted(glob.glob('results/latent_visualization/medium/*.png'))
# display first visual
from matplotlib import pyplot as plt
img = plt.imread(viz_files[0]); plt.imshow(img); plt.axis('off')
```

You can also view reconstructions:

```python
recon_files = sorted(glob.glob('results/reconstructions/hard/*.png'))
```

---

## Models & code pointers

- `src/dataset_medium.py` — multimodal loader used by `run_medium_task.py` (supports `resolve_paths_only` mode)
- `src/vae_easy.py`, `src/convae_medium.py`, `src/convae_hard.py` — model implementations
- `src/clustering_*` and `src/evaluation_*` — clustering wrappers and metric computations
- Visualization helpers: `src/visualization_hard.py`

If you need to add a new baseline or modify model hyperparameters, update the corresponding `run_*_task.py` script and re-run.

---

## Troubleshooting & tips

- Install `sentence-transformers` if medium/hard tasks fail due to missing lyrics embeddings: `pip install sentence-transformers`.
- DBSCAN often produces a single cluster on small samples — some metrics will be `NaN`; the code guards against crashes.
- If you see CUDA OOM, reduce `BATCH_SIZE` in env vars or script defaults.
- Use `SAMPLE_SIZE` for fast iterative debugging before running full experiments.

---

## Extending & tests

- To add a baseline: implement it in `src/` and call it from `run_*_task.py` scripts, adding metric computation to `evaluation_*` modules.
- Add smoke tests that run a tiny sample (`SAMPLE_SIZE=5`, `EPOCHS=1`) and assert that `results/clustering_indices_*.csv` and at least one visualization PNG exist.

---
## Project Deliverables

1. GitHub Repository with organized code
2. Implementation of Easy, Medium, and Hard tasks
3. Comprehensive evaluation metrics
4. Visualization scripts
5. NeurIPS-style paper report 

## License

This project is for educational/research purposes.

## Contact

For questions or issues, please open an issue in the repository.

