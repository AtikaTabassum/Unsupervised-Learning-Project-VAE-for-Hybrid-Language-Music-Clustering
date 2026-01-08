import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.dataset_easy import load_audio_features
from src.vae_easy import VAE
from src.clustering_easy import run_kmeans, run_pca_kmeans
from src.evaluation_easy import evaluate_clustering

CSV_PATH = "data/metadata.csv"
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '64'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
LATENT_DIM = int(os.environ.get('LATENT_DIM', '32'))
LR = float(os.environ.get('LR', '1e-3'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
PLOT_DIR = os.environ.get('PLOT_DIR', os.path.join(RESULTS_DIR, 'latent_visualization'))

PLOT_DIR_EASY = os.path.join(PLOT_DIR, 'easy')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR_EASY, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


print("Loading audio features...")
X, df = load_audio_features(CSV_PATH)
if X.shape[0] == 0:
    print("No audio features found. Make sure paths in 'data/metadata.csv' point to files under 'data/audio' and that files exist.")
    raise SystemExit(1)

SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '0'))
if SAMPLE_SIZE > 0:
    print(f"Sampling first {SAMPLE_SIZE} examples for quick run...")
    X = X[:SAMPLE_SIZE]
    df = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)

X_tensor = torch.tensor(X, dtype=torch.float32)

print("Training VAE...")
dataset = TensorDataset(X_tensor)

if len(dataset) == 0:
    print("Dataset is empty — cannot create DataLoader. Exiting.")
    raise SystemExit(1)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(input_dim=X.shape[1], latent_dim=LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def vae_loss(recon_x, x, mu, logvar):
    recon = nn.MSELoss()(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    batch_count = 0
    for (x_batch,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(x_batch)
        loss = vae_loss(recon, x_batch, mu, logvar)
      
        if not torch.isfinite(loss):
            print(f"Encountered non-finite loss (epoch {epoch+1}). Stopping training.")
            raise SystemExit(1)
        loss.backward()
     
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

print("Extracting latent representations...")
model.eval()
with torch.no_grad():
    h = model.encoder(X_tensor)
    Z = model.mu(h).cpu().numpy()

k = df["genre"].nunique()
labels_vae = run_kmeans(Z, k)
vae_metrics = evaluate_clustering(Z, labels_vae, name="VAE + KMeans")
vae_metrics["method"] = "VAE + KMeans"

Z_pca, labels_pca = run_pca_kmeans(X, k)
pca_metrics = evaluate_clustering(Z_pca, labels_pca, name="PCA + KMeans")
pca_metrics["method"] = "PCA + KMeans"

results = [vae_metrics, pca_metrics]

import pandas as pd
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "clustering_metrics_easy.csv"), index=False)

indices_df = metrics_df[['method', 'silhouette', 'calinski_harabasz']]
indices_csv = os.path.join(RESULTS_DIR, 'clustering_indices_easy.csv')
indices_df.to_csv(indices_csv, index=False)
print(f"Saved easy indices to {indices_csv}")

print("\nClustering Results:")
for r in results:
    print(r)

print("Generating t-SNE plot...")
n_samples = Z.shape[0]
perplexity = min(30, max(5, n_samples // 3))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
Z_2d = tsne.fit_transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels_vae, s=5, cmap="tab10")
plt.title("t-SNE of VAE Latent Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR_EASY, "easy_vae_tsne.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"t-SNE plot saved to {plot_path}")
print("\nEasy task completed successfully")
