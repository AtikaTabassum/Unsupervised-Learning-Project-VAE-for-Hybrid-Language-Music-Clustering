from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def run_agglomerative(X, n_clusters):
    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglom.fit_predict(X)
    return labels

def run_dbscan(X, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels
