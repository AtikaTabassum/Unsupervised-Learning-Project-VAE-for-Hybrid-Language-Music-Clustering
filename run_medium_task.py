import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from src.convae_medium import MultiModalConvVAE
from src.dataset_medium import load_multimodal_features
from src.clustering_medium import run_kmeans, run_agglomerative, run_dbscan
from src.evaluation_medium import evaluate_clustering
# PCA baseline helper
from src.clustering_easy import run_pca_kmeans


CSV_PATH = "data/metadata.csv"
BATCH_SIZE = 64
EPOCHS = 50
LATENT_DIM = 32
LR = 1e-3
RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "latent_visualization")
PLOT_DIR_MEDIUM = os.path.join(PLOT_DIR, 'medium')
# For quick testing set SAMPLE_SIZE env var (e.g., SAMPLE_SIZE=200) to limit samples
SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '0'))
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR_MEDIUM, exist_ok=True)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = torch.nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def main():
    print("Loading multi-modal features...")
    audio_feats, lyrics_feats, df = load_multimodal_features(CSV_PATH, audio_base_dir="data/audio", lyrics_base_dir="data/lyrics")


    if len(audio_feats) == 0:
        print("No valid samples found. Exiting.")
        return

    print(f"Loaded {len(audio_feats)} samples.")

    
    audio_tensor = torch.tensor(audio_feats, dtype=torch.float32)
    lyrics_tensor = torch.tensor(lyrics_feats, dtype=torch.float32)

    dataset = TensorDataset(audio_tensor, lyrics_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

  
    model = MultiModalConvVAE(lyrics_dim=lyrics_feats.shape[1], latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_audio, batch_lyrics in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_audio, batch_lyrics)
            loss = vae_loss(recon, batch_audio, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

 
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(audio_tensor, lyrics_tensor)
        Z = mu.cpu().numpy()

    
    n_clusters = int(os.environ.get('N_CLUSTERS', '10'))
    labels_true = None


    print("Running clustering algorithms...")
    clustering_results = {}

    labels_kmeans = run_kmeans(Z, n_clusters)
    clustering_results['KMeans'] = evaluate_clustering(Z, labels_kmeans, labels_true)

    labels_agglo = run_agglomerative(Z, n_clusters)
    clustering_results['Agglomerative'] = evaluate_clustering(Z, labels_agglo, labels_true)

    labels_dbscan = run_dbscan(Z, eps=0.5, min_samples=5)
    clustering_results['DBSCAN'] = evaluate_clustering(Z, labels_dbscan, labels_true)

    
    audio_feats_flat = audio_feats.reshape(len(audio_feats), -1)
    hybrid_feats = np.concatenate([audio_feats_flat, lyrics_feats], axis=1)

    Z_pca_hybrid, labels_pca_hybrid = run_pca_kmeans(hybrid_feats, n_clusters=n_clusters)
    clustering_results['PCA_hybrid'] = evaluate_clustering(Z_pca_hybrid, labels_pca_hybrid, labels_true)


    for method, scores in clustering_results.items():
        print(f"\n{method} results:")
        print(f"  Silhouette Score: {scores['silhouette']}")
        print(f"  Davies-Bouldin Index: {scores['davies_bouldin']}")
        print(f"  Adjusted Rand Index: {scores['ari']}")

    results_df = pd.DataFrame([
        {'method': m, **scores} for m, scores in clustering_results.items()
    ])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_csv = os.path.join(RESULTS_DIR, "clustering_metrics_medium.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved clustering metrics to {results_csv}")

    print("Generating t-SNE visualizations for each clustering method...")

    n_samples = Z.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    Z_tsne = tsne.fit_transform(Z)

    keys = ['KMeans', 'Agglomerative', 'DBSCAN', 'PCA_hybrid']
    indices_rows = []
    for k in keys:
        sc = clustering_results.get(k, {})
        indices_rows.append({
            'method': k,
            'silhouette': sc.get('silhouette'),
            'davies_bouldin': sc.get('davies_bouldin'),
            'ari': sc.get('ari')
        })
    indices_df = pd.DataFrame(indices_rows)
    indices_csv = os.path.join(RESULTS_DIR, 'clustering_indices_medium.csv')
    indices_df.to_csv(indices_csv, index=False)
    print(f"Saved medium indices to {indices_csv}")

    methods_labels = {
        'KMeans': labels_kmeans,
        'Agglomerative': labels_agglo,
        'DBSCAN': labels_dbscan,
        'PCA_hybrid': labels_pca_hybrid,
    }

    
    comparison_methods = [
        ('KMeans', labels_kmeans),
        ('Agglomerative', labels_agglo),
        ('DBSCAN', labels_dbscan),
        ('PCA (hybrid) + KMeans', labels_pca_hybrid),
    ]

    import math
    n_plots = len(comparison_methods)
    cols = 2
    rows = math.ceil(n_plots / cols)
    plt.figure(figsize=(cols * 6, rows * 4))
    for idx, (title, lab) in enumerate(comparison_methods):
        ax = plt.subplot(rows, cols, idx + 1)
        sc = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=lab, cmap='tab10', s=6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    comp_path = os.path.join(PLOT_DIR_MEDIUM, 'medium_cluster_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved cluster comparison to {comp_path}")

    assign_df = df.copy()
    for method, lab in methods_labels.items():
        # t-SNE plot
        plt.figure(figsize=(8, 6))
        plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=lab, cmap='tab10', s=5)
        plt.title(f"t-SNE of ConvVAE Latent Space with {method} Clusters")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR_MEDIUM, f"medium_convvae_latent_tsne_{method.lower()}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved t-SNE plot to {plot_path}")

        assign_df[f'cluster_{method.lower()}'] = lab


    clusters_csv = os.path.join(RESULTS_DIR, "clusters_medium.csv")
    assign_df.to_csv(clusters_csv, index=False)
    print(f"Saved consolidated cluster assignments to {clusters_csv}")

    print("\nMedium task completed successfully.")

if __name__ == "__main__":
    main()
