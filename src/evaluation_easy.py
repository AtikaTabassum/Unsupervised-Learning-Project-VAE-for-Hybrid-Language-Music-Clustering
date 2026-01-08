from sklearn.metrics import silhouette_score, calinski_harabasz_score


def evaluate_clustering(X, labels, name="Model"):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    print(f"{name} Evaluation:")
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  Calinski-Harabasz Index: {ch:.2f}")

    return {
        "silhouette": sil,
        "calinski_harabasz": ch
    }
