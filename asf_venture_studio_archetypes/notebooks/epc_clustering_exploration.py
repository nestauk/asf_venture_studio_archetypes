# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Clustering exploration

# %%
# cd ~/asf_venture_studio_archetypes/

# %% [markdown]
# ## OPTION A
# - one-hot encode nominal features
# - DBSCAN

# %%
import pandas as pd
import numpy as np
from asf_venture_studio_archetypes.pipeline import epc_processing
from asf_venture_studio_archetypes.config import base_epc
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# %%
dim_red_data = pd.read_csv("outputs/data/epc_dim_reduced_data.csv").drop(
    "Unnamed: 0", axis=1
)

# %%
# Apply one-hot encoding
encoded_data = epc_processing.one_hot_encoding(
    dim_red_data, cols=base_epc.EPC_FEAT_NOMINAL
)

# %%
# perform DBSCAN
db = DBSCAN(eps=3, min_samples=2)

# Fit the model to the data
db.fit(encoded_data)

# Get the cluster assignments for each data point
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

silhouette_score = silhouette_score(encoded_data, labels)

# Print the silhouette score
print("Silhouette Score:", silhouette_score)


# %%
## Iterate over a set of eps and min_sample parameters

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Define a set of epsilon values to try
eps_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

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

# Plot the silhouette scores as a heatmap
plt.imshow(silhouette_scores, cmap="hot", origin="lower")
plt.xticks(np.arange(len(min_samples_vals)), min_samples_vals)
plt.yticks(np.arange(len(eps_vals)), eps_vals)
plt.xlabel("min_samples")
plt.ylabel("eps")
plt.colorbar()
plt.show()


# %% [markdown]
# ## OPTION B
# - encode nominal features
# - Use K-means with elbow method

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def elbow_method(X, max_clusters):
    """
    Implements the elbow method to determine the optimal number of clusters
    for a KMeans algorithm.

    Parameters:
    - X: numpy array, data to be clustered
    - max_clusters: int, maximum number of clusters to consider

    Returns:
    - optimal_clusters: int, optimal number of clusters
    """
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


# Example usage
optimal_clusters = elbow_method(encoded_data.values, 50)
print("Optimal number of clusters:", optimal_clusters)


# %%
dim_red_data[base_epc.EPC_FEAT_NOMINAL]

# %%
base_epc.EPC_FEAT_ORDINAL

# %% [markdown]
# # Visualise clustering

# %%
dim_red_data

# %%
import seaborn as sns

scat = sns.scatterplot(
    data=dim_red_data.sample(100000), x="PC0", y="PC1", hue="BUILT_FORM", alpha=0.4
)
fig = scat.get_figure()
fig.savefig("scat.png")

# %%
dim_red_data.columns

# %%
