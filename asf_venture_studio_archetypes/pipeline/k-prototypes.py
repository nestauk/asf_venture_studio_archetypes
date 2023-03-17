import sys
import os
import sys
import asf_core_data


from asf_core_data import Path
from asf_venture_studio_archetypes.utils.epc_processing import *
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.utils.feature_engineering import *

from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import altair as alt
from datetime import datetime

# yyy-mm-dd
date = datetime.today().strftime("%Y-%m-%d")
dimension = 15
num_feat = base_epc.EPC_FEAT_NUM_KPROTO
cat_feat = base_epc.EPC_FEAT_CAT_KPROTO
all_feat = base_epc.EPC_FEAT_ALL_KPROTO
n_sample = 5000

prep_epc = load_data(all_feat, n_sample)
# processed_data = process_data(prep_epc, num_feat, cat_feat, oh_encoder=True)
# prep_epc = load_data(all_feat, n_sample)
# original_categories = process_data(prep_epc, num_feat, cat_feat)
# prep_epc = load_data(all_feat, n_sample)


def KProto_apply(
    df: pd.DataFrame = prep_epc,
    dimension: int = dimension,
    num_feat: list = num_feat,
    all_feat: list = all_feat,
    n_sample=n_sample,
    n_clusters=4,
    n_neighbors=30,
    min_dist=0.0,
):
    start_time = time.time()
    print("Performing K-prototypes.")

    processed_data = process_data(prep_epc.copy(), num_feat, cat_feat, oh_encoder=True)
    original_categories = process_data(prep_epc.copy(), num_feat, cat_feat)

    numerical = processed_data[num_feat]
    categorical = processed_data.drop(columns=num_feat)

    # Percentage of columns which are categorical is used as weight parameter in embeddings later
    categorical_weight = (
        len(processed_data.select_dtypes(include="object").columns)
        / processed_data.shape[1]
    )

    # Embedding numerical & categorical
    fit1 = umap.UMAP(
        metric="l2",
        n_components=dimension,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    ).fit(numerical)
    fit2 = umap.UMAP(
        metric="dice",
        n_components=dimension,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    ).fit(categorical)

    # Augmenting the numerical embedding with categorical
    intersection = umap.umap_.general_simplicial_set_intersection(
        fit1.graph_, fit2.graph_, weight=categorical_weight
    )
    intersection = umap.umap_.reset_local_connectivity(intersection)
    densmap_kwds = {}
    embedding = umap.umap_.simplicial_set_embedding(
        fit1._raw_data,
        intersection,
        fit1.n_components,
        fit1._initial_alpha,
        fit1._a,
        fit1._b,
        fit1.repulsion_strength,
        fit1.negative_sample_rate,
        200,
        "random",
        np.random,
        fit1.metric,
        fit1._metric_kwds,
        densmap_kwds=densmap_kwds,
        output_dens=False,
        densmap=False,
    )

    embedding = embedding[0]

    catColumnsPos = [
        processed_data.columns.get_loc(col)
        for col in list(processed_data.drop(columns=num_feat))
    ]

    # Actual clustering
    kprot_data = processed_data.copy()
    kproto = KPrototypes(n_clusters=n_clusters, init="Cao", n_jobs=4, random_state=42)
    cluster = kproto.fit_predict(kprot_data, categorical=catColumnsPos)

    processed_data["x"] = embedding[:, 0]
    processed_data["y"] = embedding[:, 1]
    processed_data["clusters"] = list(cluster)
    ohe_encoded_data = processed_data.copy()

    original_categories["x"] = embedding[:, 0]
    original_categories["y"] = embedding[:, 1]
    original_categories["clusters"] = list(cluster)

    plt.scatter(x=embedding[:, 0], y=embedding[:, 1])
    plt.savefig(f"outputs/figures/kproto-{date}-scatter-{n_neighbors}-{min_dist}.png")
    processed_data.to_csv(
        f"outputs/data/kproto-{date}-ohe_catgories-{n_neighbors}-{min_dist}.csv"
    )
    original_categories.to_csv(
        f"outputs/data/kproto-{date}-original_categories-{n_neighbors}-{min_dist}.csv"
    )

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("K-prototypes took {} minutes.\n".format(runtime))

    return kproto


def KProto_elbow_method(
    kprot_data_date="2023-03-15", num_feat=num_feat, max_cluster=16
):
    kprot_data = pd.read_csv(
        f"outputs/data/kproto-{kprot_data_date}-ohe_catgories.csv"
    ).copy()

    costs = []
    n_clusters = []
    clusters_assigned = []

    catColumnsPos = [
        kprot_data.columns.get_loc(col)
        for col in list(kprot_data.drop(columns=num_feat))
    ]

    for i in tqdm(range(2, max_cluster)):
        try:
            kproto = KPrototypes(n_clusters=i, init="Cao", verbose=2)
            clusters = kproto.fit_predict(kprot_data, categorical=catColumnsPos)
            costs.append(kproto.cost_)
            n_clusters.append(i)
            clusters_assigned.append(cluster)
        except:
            print(f"Can't cluster with {i} clusters")

    plt.scatter(x=n_clusters, y=costs)
    plt.savefig(
        f"outputs/figures/kproto-elbow-method-{kprot_data_date}-{max_cluster}clusters.png"
    )


def KProto_silhouette_analysis(
    n_cluster_vals, plot: bool = True, num_feat=num_feat
) -> int:
    """Perform Silhouette analysis with K-means to determine optimal number of cluster

    Args:
        df (pd.DataFrame): Data to use for clustering (numerical)
        n_cluster_vals (Iterator): Range of cluster numners to use of the S-analysis
        plot (bool, optional): Plot and save plot. Defaults to True.

    Returns:
        int: Optimal number of clusters.
    """

    start_time = time.time()
    print("Performing Silhouette analysis.")

    prep_epc = load_data(all_feat, n_sample)
    df = process_data(prep_epc, num_feat, cat_feat, oh_encoder=True)

    X = df.values
    sil_coeff = np.zeros(np.shape(n_cluster_vals))
    for i, n_cluster in enumerate(n_cluster_vals):
        kproto = KProto_apply(n_clusters=n_cluster).fit(X)
        print(f"{kproto}")
        catColumnsPos = [
            df.columns.get_loc(col) for col in list(df.drop(columns=num_feat))
        ]
        print(f"{catColumnsPos}")
        label = kproto.fit_predict(df, categorical=catColumnsPos)
        sil_coeff[i] = silhouette_score(X, label, metric="euclidean")

    if plot:
        plt.plot(n_cluster_vals, sil_coeff)
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.savefig("outputs/figures/silhouette_analysis.png")

    print(
        "Silhouette Analysis: optimal number of clusters: {} with Silhouette Score of {}".format(
            list(n_cluster_vals)[list(sil_coeff).index(max(sil_coeff))], max(sil_coeff)
        )
    )

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Silhouette analysis took {} minutes.\n".format(runtime))

    return list(n_cluster_vals)[list(sil_coeff).index(max(sil_coeff))]


if __name__ == "__main__":
    KProto_apply()
    # KProto_silhouette_analysis(n_cluster_vals = [2,3,4])
