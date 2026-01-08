from src.convae_hard import MultiModalBetaVAE
from src.dataset_medium import load_multimodal_features
from src.clustering_hard import run_kmeans
from src.evaluation_hard import evaluate_clustering
from src.visualization_hard import visualize_latent, plot_cluster_distribution, save_reconstructions
import torch
import numpy as np
import os

CSV_PATH = os.environ.get('CSV_PATH', 'data/metadata.csv')
RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'latent_visualization')
PLOT_DIR_HARD = os.path.join(PLOT_DIR, 'hard')

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR_HARD, exist_ok=True)

def main():
    RESOLVE_PATHS_ONLY = os.environ.get('RESOLVE_PATHS_ONLY', '0') == '1'
    audio, lyrics, df = load_multimodal_features(CSV_PATH, resolve_paths_only=RESOLVE_PATHS_ONLY)

    if len(audio) == 0:
        print('No samples found for hard task. Check CSV and data dirs.')
        return

    if RESOLVE_PATHS_ONLY:
        print(f"Resolved {len(audio)} samples (paths only). Example audio path: {audio[0]}, lyrics path: {lyrics[0]}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio = torch.tensor(audio).float().to(device)
    lyrics = torch.tensor(lyrics).float().to(device)
   
    if 'genre' in df.columns:
        genres = df['genre'].astype('category').cat.codes.values
        num_genres = df['genre'].nunique()
        genre_names = list(df['genre'].astype('category').cat.categories)
    else:
        genres = None
        num_genres = None
        genre_names = None

    SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '0'))
    EPOCHS = int(os.environ.get('EPOCHS', '50'))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '64'))
    LR = float(os.environ.get('LR', '1e-3'))
    BETA = float(os.environ.get('BETA', '4.0'))

    if SAMPLE_SIZE > 0:
        audio = audio[:SAMPLE_SIZE]
        lyrics = lyrics[:SAMPLE_SIZE]
        genres = genres[:SAMPLE_SIZE] if genres is not None else None
        df = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)

    def train_model(model, audio_t, lyrics_t, genres_t=None, epochs=30, batch_size=64, lr=1e-3):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(audio_t, lyrics_t) if genres_t is None else torch.utils.data.TensorDataset(audio_t, lyrics_t, torch.tensor(genres_t, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            total = 0.0
            cnt = 0
            for batch in loader:
                if genres_t is None:
                    x_b, l_b = batch
                    g_b = None
                else:
                    x_b, l_b, g_b = batch
                optimizer.zero_grad()
                recon, mu, logvar = model(x_b, l_b, genre=g_b if g_b is not None else None)
                loss = model.loss(recon, x_b, mu, logvar)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss encountered during training')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total += loss.item()
                cnt += 1
            print(f"Train epoch {epoch+1}/{epochs} - loss {total/cnt:.4f}")
        return model

    model_vae = MultiModalBetaVAE(lyrics_dim=lyrics.shape[1], latent_dim=32, beta=BETA, num_genres=num_genres if num_genres else None).to(device)
    model_vae = train_model(model_vae, audio, lyrics, genres if num_genres else None, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    model_vae.eval()
    with torch.no_grad():
        mu, _ = model_vae.encode(audio, lyrics, genre=torch.tensor(genres, dtype=torch.long).to(device) if num_genres else None)
    z_vae = mu.cpu().numpy()

    labels_vae = run_kmeans(z_vae, n_clusters=df['genre'].nunique() if 'genre' in df.columns else 10)

    from src.clustering_easy import run_pca_kmeans
    Z_pca, labels_pca = run_pca_kmeans(z_vae, n_clusters=df['genre'].nunique() if 'genre' in df.columns else 10)

    model_ae = MultiModalBetaVAE(lyrics_dim=lyrics.shape[1], latent_dim=32, beta=0.0, num_genres=num_genres if num_genres else None).to(device)
    model_ae = train_model(model_ae, audio, lyrics, genres if num_genres else None, epochs=max(1, EPOCHS//2), batch_size=BATCH_SIZE, lr=LR)
    model_ae.eval()
    with torch.no_grad():
        mu_ae, _ = model_ae.encode(audio, lyrics, genre=torch.tensor(genres, dtype=torch.long).to(device) if num_genres else None)
    z_ae = mu_ae.cpu().numpy()
    labels_ae = run_kmeans(z_ae, n_clusters=df['genre'].nunique() if 'genre' in df.columns else 10)

    audio_np = audio.cpu().numpy()
    audio_flat = audio_np.reshape(len(audio_np), -1)
    Z_pca_spec, labels_pca_spec = run_pca_kmeans(audio_flat, n_clusters=df['genre'].nunique() if 'genre' in df.columns else 10)

    labels_spec_kmeans = run_kmeans(audio_flat, n_clusters=df['genre'].nunique() if 'genre' in df.columns else 10)

    true_labels = df['genre'].astype('category').cat.codes.values if 'genre' in df.columns else None
  
    metrics = []
    m_vae = evaluate_clustering(z_vae, labels_vae, labels_true=true_labels)
    m_vae['method'] = 'VAE + KMeans'
    metrics.append(m_vae)


    m_ae = evaluate_clustering(z_ae, labels_ae, labels_true=true_labels)
    m_ae['method'] = 'AE + KMeans'
    metrics.append(m_ae)

    m_pca = evaluate_clustering(Z_pca, labels_pca, labels_true=true_labels)
    m_pca['method'] = 'PCA + KMeans'
    metrics.append(m_pca)

    m_pca_spec = evaluate_clustering(Z_pca_spec, labels_pca_spec, labels_true=true_labels)
    m_pca_spec['method'] = 'PCA (spectral) + KMeans'
    metrics.append(m_pca_spec)

    m_spec_kmeans = evaluate_clustering(audio_flat, labels_spec_kmeans, labels_true=true_labels)
    m_spec_kmeans['method'] = 'Spectral KMeans'
    metrics.append(m_spec_kmeans)

    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.join(RESULTS_DIR, 'clustering_metrics_hard.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved clustering metrics to {metrics_csv}")

    indices_methods = ['VAE + KMeans', 'PCA (spectral) + KMeans', 'AE + KMeans', 'Spectral KMeans']
    indices_rows = []
    for m in indices_methods:
        entry = next((d for d in metrics if d.get('method') == m), {})
        indices_rows.append({
            'method': m,
            'silhouette': entry.get('silhouette'),
            'nmi': entry.get('nmi'),
            'ari': entry.get('ari'),
            'purity': entry.get('purity')
        })
    import pandas as pd
    indices_df = pd.DataFrame(indices_rows)
    indices_csv = os.path.join(RESULTS_DIR, 'clustering_indices_hard.csv')
    indices_df.to_csv(indices_csv, index=False)
    print(f"Saved hard indices to {indices_csv}")

    visualize_latent(z_vae, labels_vae, os.path.join(PLOT_DIR_HARD, "hard_vae_kmeans.png"))
    if true_labels is not None:
        plot_cluster_distribution(labels_vae, true_labels, genre_names=genre_names, save_path=os.path.join(PLOT_DIR_HARD, 'hard_vae_cluster_dist.png'))

    with torch.no_grad():
        recon, _, _ = model_vae.forward(audio, lyrics, genre=torch.tensor(genres, dtype=torch.long).to(device) if num_genres else None)
    from src.visualization_hard import save_reconstructions
    save_reconstructions(audio.cpu().numpy(), recon.cpu().numpy(), out_dir=os.path.join(RESULTS_DIR, 'reconstructions', 'hard'), n=8, prefix='hard_')

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        Z_tsne = tsne.fit_transform(z_vae)
    except Exception:
        Z_tsne = None

    comparison_methods = [
        ('VAE + KMeans', labels_vae),
        ('PCA (spectral) + KMeans', labels_pca_spec),
        ('AE + KMeans', labels_ae),
        ('Spectral KMeans', labels_spec_kmeans),
    ]

    import math
    import matplotlib.pyplot as plt
    n_plots = len(comparison_methods)
    cols = 2
    rows = math.ceil(n_plots / cols)
    plt.figure(figsize=(cols * 6, rows * 4))
    for idx, (title, lab) in enumerate(comparison_methods):
        ax = plt.subplot(rows, cols, idx + 1)
        if Z_tsne is not None:
            ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=lab, cmap='tab10', s=6)
        else:
            ax.scatter(z_vae[:, 0], z_vae[:, 1], c=lab, cmap='tab10', s=6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    comp_path = os.path.join(PLOT_DIR_HARD, 'hard_cluster_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved hard cluster comparison to {comp_path}")

    assign_df = df.copy()
    assign_df['cluster_vae'] = labels_vae
    assign_df['cluster_ae'] = labels_ae
    assign_df['cluster_pca_spectral'] = labels_pca_spec
    assign_df['cluster_spec_kmeans'] = labels_spec_kmeans
    clusters_csv = os.path.join(RESULTS_DIR, 'clusters_hard.csv')
    assign_df.to_csv(clusters_csv, index=False)
    print(f"Saved cluster assignments to {clusters_csv}")

if __name__ == "__main__":
    main()
