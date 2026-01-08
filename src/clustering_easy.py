from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_kmeans(X, k):
    return KMeans(n_clusters=k, random_state=42).fit_predict(X)


def run_pca_kmeans(X, n_clusters, n_components=32):
    n_samples, n_features = X.shape
    n_components = max(1, min(n_components, n_samples - 1, n_features))
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Z)
    return Z, labels
