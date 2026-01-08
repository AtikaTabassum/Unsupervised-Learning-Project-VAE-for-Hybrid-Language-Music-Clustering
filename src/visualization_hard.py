import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_latent(z, labels, save_path, title="Latent Space (t-SNE)", method='tsne'):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            z_2d = reducer.fit_transform(z)
        except Exception:
            tsne = TSNE(n_components=2, random_state=42)
            z_2d = tsne.fit_transform(z)
    else:
        n = len(z)
        perplexity = min(30, max(5, n // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return z_2d


def plot_cluster_distribution(labels, genres, genre_names=None, save_path=None, title="Cluster distribution by genre"):
    labels = np.asarray(labels)
    genres = np.asarray(genres)
    clusters = np.unique(labels)
    genres_unique = np.unique(genres)

    counts = np.zeros((len(clusters), len(genres_unique)), dtype=int)
    for i, c in enumerate(clusters):
        for j, g in enumerate(genres_unique):
            counts[i, j] = np.sum((labels == c) & (genres == g))

   
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    bottoms = np.zeros(len(clusters))
    for j, g in enumerate(genres_unique):
        vals = counts[:, j]
        label = genre_names[g] if (genre_names is not None and g < len(genre_names)) else str(g)
        plt.bar(clusters, vals, bottom=bottoms, label=label)
        bottoms += vals

    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def save_reconstructions(original_specs, recon_specs, out_dir, n=8, prefix=''):
   
    os.makedirs(out_dir, exist_ok=True)
    N = min(n, len(original_specs))
    import matplotlib.pyplot as plt
    for i in range(N):
        orig = original_specs[i]
        recon = recon_specs[i]
        if orig.ndim == 3:
            orig = orig.squeeze(0)
        if recon.ndim == 3:
            recon = recon.squeeze(0)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(orig, aspect='auto', origin='lower')
        axes[0].set_title('Original')
        axes[1].imshow(recon, aspect='auto', origin='lower')
        axes[1].set_title('Reconstruction')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'{prefix}recon_{i}.png')
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
