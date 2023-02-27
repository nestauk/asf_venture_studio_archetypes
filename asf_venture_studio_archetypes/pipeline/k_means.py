from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def KMeans_apply(df: pd.DataFrame, col_name: str = "cluster", **kwargs) -> pd.DataFrame:
    """Apply K-means and add labels to DataFrame

    Args:
        df (pd.DataFrame): Data to perform K-means on.
        kwargs (_type_): kwargs to KMeans
        col_name (str, optional): Column name of cluster labels. Defaults to "cluster".

    Returns:
        pd.DataFrame: Data with clustering labels annotations
    """

    X = df.values
    kmeans = KMeans(kwargs).fit(X)
    labels = kmeans.labels_
    df[col_name] = labels


def KMeans_elbow_method(df: pd.DataFrame, max_clusters: int) -> int:
    """Perform elbow method with K-means to determine optimal number of cluster

    TO DO: add visualisation plot

    Args:
        data (pd.DataFrame): Data to use for clustering (numerical)
        max_clusters (int): Maximum number of clusters to use for K-means

    Returns:
        int: Optimal number of clusters. Returns None of not found.
    """

    # Exctract values
    X = df.values
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters + 1), inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.show()

    # Determine the optimal number of clusters using the elbow method
    optimal_clusters = None
    for i in range(1, len(inertias) - 1):
        if inertias[i] < inertias[i - 1] and inertias[i] < inertias[i + 1]:
            optimal_clusters = i + 1
            break

    return optimal_clusters


def KMeans_silhouette_analysis(
    df: pd.DataFrame, n_cluster_vals: Iterator, plot: bool = True
) -> int:
    """Perform Silhouette analysis with K-means to determine optimal number of cluster

    Args:
        df (pd.DataFrame): Data to use for clustering (numerical)
        n_cluster_vals (Iterator): Range of cluster numners to use of the S-analysis
        plot (bool, optional): Plot and save plot. Defaults to True.

    Returns:
        int: Optimal number of clusters.
    """
    X = df.values
    sil_coeff = np.zeros(np.shape(n_cluster_vals))
    for i, n_cluster in enumerate(n_cluster_vals):
        kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(X)
        label = kmeans.labels_
        sil_coeff[i] = silhouette_score(X, label, metric="euclidean")

    if plot:
        plt.plot(n_cluster_vals, sil_coeff)
        plt.savefig("outputs/figures/silhouette_analysis.png")

    print(
        "Silhouette Analysis: optimal number of clusters: {} with Silhouette Score of {}".format(
            list(n_cluster_vals)[list(sil_coeff).index(max(sil_coeff))], max(sil_coeff)
        )
    )

    return list(n_cluster_vals)[list(sil_coeff).index(max(sil_coeff))]


# if __name__ == "__main__":
#     # Execute only if run as a script
