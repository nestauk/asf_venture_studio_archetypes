from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from asf_venture_studio_archetypes.utils.epc_processing import *
from asf_venture_studio_archetypes.config import base_epc
from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def KMeans_apply(df: pd.DataFrame, col_name: str = "cluster", **kwargs) -> pd.DataFrame:
    """Apply K-means and add labels to DataFrame

    Args:
        df (pd.DataFrame): Data to perform K-means on.
        kwargs (_type_): kwargs to KMeans
        col_name (str, optional): Column name of cluster labels. Defaults to "cluster".

    Returns:
        pd.DataFrame: Data with clustering labels annotations
    """
    start_time = time.time()
    print("Performing K-means.")
    X = df.values
    kmeans = KMeans(**kwargs).fit(X)
    labels = kmeans.labels_
    df[col_name] = labels

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("K-means took {} minutes.\n".format(runtime))
    return df


def KMeans_elbow_method(df: pd.DataFrame, max_clusters: int, plot: bool = True) -> int:
    """Perform elbow method with K-means to determine optimal number of cluster

    TO DO: add visualisation plot

    Args:
        data (pd.DataFrame): Data to use for clustering (numerical)
        max_clusters (int): Maximum number of clusters to use for K-means
        plot (bool): Plot and save results of elbow analysis. Default to True.

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

    if plot:
        plt.plot(range(1, max_clusters + 1), inertias, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.savefig("outputs/figures/elbow_analysis.png")

    # Determine the optimal number of clusters using the elbow method
    optimal_clusters = None
    for i in range(1, len(inertias) - 1):
        if inertias[i] < inertias[i - 1] and inertias[i] < inertias[i + 1]:
            optimal_clusters = i + 1
            break
    return optimal_clusters


def KMeans_silhouette_analysis(
    df: pd.DataFrame, n_cluster_vals: Iterator, plot: bool = True, reps: int = 1
) -> int:
    """Perform Silhouette analysis with K-means to determine optimal number of cluster

    Args:
        df (pd.DataFrame): Data to use for clustering (numerical)
        n_cluster_vals (Iterator): Range of cluster numners to use of the S-analysis
        plot (bool, optional): Plot and save plot. Defaults to True.
        reps (int, optional): Repeat silhouette analysis rep times and avarege results.

    Returns:
        int: Optimal number of clusters.
    """

    start_time = time.time()
    print("Performing Silhouette analysis.")

    X = df.values
    sil_coeff = np.zeros((reps, len(n_cluster_vals)))

    for rep in range(reps):
        for i, n_cluster in enumerate(n_cluster_vals):
            kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(X)
            label = kmeans.labels_
            sil_coeff[rep, i] = silhouette_score(X, label, metric="euclidean")

    if plot:
        sns.lineplot(pd.DataFrame(sil_coeff).melt(), x="variable", y="value")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.savefig("outputs/figures/silhouette_analysis.png")

    if reps > 1:
        sil_coeff = np.mean(sil_coeff)
    else:
        sil_coeff = np.array(sil_coeff)

    print(
        "Silhouette Analysis: optimal number of clusters: {} with Silhouette Score of {}".format(
            np.argmax(sil_coeff), np.max(sil_coeff)
        )
    )

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Silhouette analysis took {} minutes.\n".format(runtime))

    return np.argmax(sil_coeff)


def pipeline_kmeans_selected_features():
    """Pipeline that apply kmeans on selected features"""

    feat_list_num = base_epc.EPC_FEAT_NUM_KMEANS
    feat_list_cat = base_epc.EPC_FEAT_CAT_KMEANS
    n_sample = 10000

    feat_list = feat_list_num + feat_list_cat

    # Loading and processing
    processed_data = load_data(feat_list, n_sample)
    processed_data = process_data(
        processed_data, feat_list_num, feat_list_cat, oh_encoder=True
    )

    # Clustering
    processed_data = KMeans_apply(processed_data, n_clusters=3, n_init=10)

    # Saving data
    processed_data.to_csv("outputs/data/k_means_selected_features.csv")

    return processed_data


if __name__ == "__main__":
    # Execute only if run as a script
    pipeline_kmeans_selected_features()
