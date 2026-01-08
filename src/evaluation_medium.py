from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def evaluate_clustering(X, labels_pred, labels_true=None):
    scores = {}
    if len(set(labels_pred)) > 1:
        scores['silhouette'] = silhouette_score(X, labels_pred)
        scores['davies_bouldin'] = davies_bouldin_score(X, labels_pred)
    else:
        scores['silhouette'] = None
        scores['davies_bouldin'] = None

    if labels_true is not None:
        scores['ari'] = adjusted_rand_score(labels_true, labels_pred)
    else:
        scores['ari'] = None

    return scores
