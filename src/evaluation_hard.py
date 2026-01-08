from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
import numpy as np


def purity_score(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def evaluate_clustering(z, labels, labels_true=None):
    scores = {
        "silhouette": None,
        "davies_bouldin": None,
        "nmi": None,
        "ari": None,
        "purity": None
    }

    unique_labels = len(set(labels)) if labels is not None else 0
    if unique_labels > 1:
        scores["silhouette"] = silhouette_score(z, labels)
        scores["davies_bouldin"] = davies_bouldin_score(z, labels)

    if labels_true is not None:
        scores["nmi"] = normalized_mutual_info_score(labels_true, labels)
        scores["ari"] = adjusted_rand_score(labels_true, labels)
        scores["purity"] = purity_score(labels_true, labels)

    return scores
