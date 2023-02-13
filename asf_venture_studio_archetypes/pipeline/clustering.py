import pandas as pd
import numpy as np
from asf_venture_studio_archetypes.pipeline import epc_processing
from asf_venture_studio_archetypes.config import base_epc
from typing import Union, List, Type
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def dbscan(data: pd.DataFrame, eps: float = 3, min_sample: int = 2) -> dict:
    """Perform DBSCAN on data (numerical) and return number of cluster,
    numer of noise points and silhoutte score.

    Args:
        data (pd.DataFrame): Data to use for the clustering
        eps (float, optional): eps parameter of dbscan. Defaults to 3.
        min_sample (int, optional): min_sample parameter of dbscan. Defaults to 2.

    Returns:
        dict: Dictionary with "n_clusters", "n_noise" and "silhoutte_score".
    """

    # perform DBSCAN
    db = DBSCAN(eps=3, min_samples=2)

    # Fit the model to the data
    db.fit(data)

    # Get the cluster assignments for each data point
    labels = dbscan.labels_

    return {
        "n_cluster": len(set(labels)) - (1 if -1 in labels else 0),
        "n_noise": list(labels).count(-1),
        "silhouette_score": silhouette_score(encoded_data, labels),
    }


def elbow_method(data: pd.DataFrame, max_clusters: int) -> int:
    """Perform elbow method with K-means to determine optimal number of cluster

    Args:
        data (pd.DataFrame): Data to use for clustering (numerical)
        max_clusters (int): Maximum number of clusters to use for K-means

    Returns:
        int: Optimal number of clusters. Returns None of not found.
    """

    # Exctract values
    X = data.values
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


def dbscan_grid_search(
    df: pd.DataFrame,
    eps_vals: List[float] = None,
    min_samples_vals: List[int] = None,
    save_plot: bool = True,
):
    """Perform dbscan with grid search parameters for eps and min_samples

    Args:
        df (pd.DataFrame): Data to use for clsutering
        eps_vals (List[int], optional): List of values of eps parameter
        min_samples_vals (List[int]): List fo values for min_sample parameter
        save_plot (bool, optional): Option to save the results of the grid search as heatmap. Defaults to True.
    """
    # Apply one-hot encoding
    encoded_data = epc_processing.one_hot_encoding(df, cols=base_epc.EPC_FEAT_NOMINAL)

    if not eps_vals:
        # Define a set of epsilon values to try
        eps_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    if not min_samples_vals:
        # Define a set of min_samples values to try
        min_samples_vals = [2, 3, 4, 5, 6, 7, 8, 9]

    # Initialize an array to store the silhouette scores
    silhouette_scores = np.zeros((len(eps_vals), len(min_samples_vals)))

    # Iterate over the combinations of epsilon and min_samples values
    for i, eps in enumerate(eps_vals):
        for j, min_samples in enumerate(min_samples_vals):
            # Initialize the DBSCAN model with the current combination of parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)

            # Fit the model to the data
            dbscan.fit(encoded_data)

            # Get the cluster assignments for each data point
            labels = dbscan.labels_

            # Calculate the silhouette score for the current combination of parameters
            silhouette_scores[i, j] = silhouette_score(encoded_data, labels)

    if save_plot:
        # Plot the silhouette scores as a heatmap
        plt.imshow(silhouette_scores, cmap="hot", origin="lower")
        plt.xticks(np.arange(len(min_samples_vals)), min_samples_vals)
        plt.yticks(np.arange(len(eps_vals)), eps_vals)
        plt.xlabel("min_samples")
        plt.ylabel("eps")
        plt.colorbar()
        plt.savefig("outputs/figures/dbscan_grid_search.png")


if __name__ == "__main__":
    # Load dimensionality reduced data
    dim_red_data = pd.read_csv("outputs/data/epc_dim_reduced_data.csv").drop(
        "Unnamed: 0", axis=1
    )
    dbscan_grid_search(dim_red_data)
