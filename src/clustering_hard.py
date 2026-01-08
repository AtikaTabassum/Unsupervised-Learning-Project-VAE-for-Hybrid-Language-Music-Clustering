import numpy as np
from sklearn.cluster import KMeans



def run_kmeans(z, n_clusters=10):
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(z)



def run_kmeans_with_genre(z, genres, n_clusters=10, num_genres=None, genre_weight=1.0):
    
    if genres is None or num_genres is None:
        return run_kmeans(z, n_clusters=n_clusters)
    genres = np.asarray(genres)
    one_hot = np.zeros((len(genres), num_genres), dtype=float)
    one_hot[np.arange(len(genres)), genres.astype(int)] = 1.0
    aug = np.concatenate([z, one_hot * genre_weight], axis=1)
    return run_kmeans(aug, n_clusters=n_clusters)


