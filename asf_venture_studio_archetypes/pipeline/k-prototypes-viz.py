import sys
import os
import sys
import asf_core_data
import pandas as pd
from asf_core_data import load_preprocessed_epc_data
from asf_core_data.getters.epc import epc_data, data_batches
from asf_venture_studio_archetypes.utils.plotting import configure_plots
from asf_core_data import Path
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.epc_processing import *
from asf_venture_studio_archetypes.pipeline.dimensionality_reduction import (
    load_and_process_data,
)
from sklearn.preprocessing import PowerTransformer
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

# yyy-mm-dd
date = datetime.today().year
dimension = 15


# define local data directory
DATA_DIR = "/Users/imanmuse/Documents/data/EPC"

# Load preprocessed epc data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path=DATA_DIR,
    version="preprocessed_dedupl",
    usecols=base_epc.EPC_PREP_CLEAN_USE_FEAT_SELECTION,
    batch="newest",
    n_samples=5000,  # Comment to run on full dataset (~40 min)
)

# # Further data cleaning
# feat_drop = ["POSTCODE"]  # haven't decided how to handle it yet
# prep_epc.drop(columns=feat_drop, inplace=True)

# # Extract year of inspection date
# prep_epc = extract_year_inspection(prep_epc)


full_data = load_and_process_data()

num_feat = list(
    full_data.columns.intersection(
        base_epc.EPC_PREP_NUMERICAL + base_epc.EPC_PREP_ORDINAL
    )
)
numerical = full_data[num_feat]

# Transform categorical features
cat_feat = list(prep_epc.columns.intersection(base_epc.EPC_PREP_CATEGORICAL))

# One hot encoding
encoded_features = one_hot_encoding(prep_epc, cat_feat)
categorical = encoded_features

# Percentage of columns which are categorical is used as weight parameter in embeddings later
categorical_weight = (
    len(full_data.select_dtypes(include="object").columns) / full_data.shape[1]
)

# Embedding numerical & categorical
fit1 = umap.UMAP(metric="l2", n_components=dimension).fit(numerical)
fit2 = umap.UMAP(metric="dice", n_components=dimension).fit(categorical)

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

plt.figure(figsize=(20, 10))
embedding = embedding[0]
# plt.scatter(*embedding.T, s=2, cmap="Spectral", alpha=1.0)
# plt.show()


kprot_data = full_data.copy()
# Pre-processing
for c in full_data.select_dtypes(exclude="object").columns:
    pt = PowerTransformer()
    kprot_data[c] = pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

# Actual clustering
kproto = KPrototypes(n_clusters=15, init="Cao", n_jobs=4)
clusters = kproto.fit_predict(
    kprot_data, categorical=list(range(18, len(full_data.columns)))
)

# Prints the count of each cluster group
pd.Series(clusters).value_counts()

# OPTIONAL: Elbow plot with cost (will take a LONG time)
costs = []
n_clusters = []
clusters_assigned = []

# for i in tqdm(range(2, 25)):
#     try:
#         kproto = KPrototypes(n_clusters= i, init='Cao', verbose=2)
#         clusters = kproto.fit_predict(kprot_data, categorical=list(range(18,len(full_data.columns))))
#         costs.append(kproto.cost_)
#         n_clusters.append(i)
#         clusters_assigned.append(clusters)
#     except:
#         print(f"Can't cluster with {i} clusters")

# costs = 4
# n_clusters = 15

# fig = go.Figure(data=go.Scatter(x=n_clusters, y=costs ))
# fig.show()

# fig, ax = plt.subplots()
# fig.set_size_inches((20, 10))
# scatter = ax.scatter(
#     embedding[:, 0], embedding[:, 1], s=2, c=clusters, cmap="tab20b", alpha=1.0
# )

# # produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(num=15), loc="lower left", title="Classes")
# ax.add_artist(legend1)

# scatter plot
prep_epc["x"] = embedding[:, 0]
prep_epc["y"] = embedding[:, 1]
prep_epc["clusters"] = list(clusters)

fig = (
    alt.Chart(prep_epc)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color="clusters",
        tooltip=[
            "CONSTRUCTION_AGE_BAND",
            "CURRENT_ENERGY_EFFICIENCY",
            "CO2_EMISSIONS_CURRENT",
            "TENURE",
        ],
    )
    .interactive()
)

fig.save(f"outputs/figures/k-proto-scatter-{date}-{dimension}-dimension.html")
prep_epc.to_csv(f"outputs/data/k-proto-cluster-{date}-{dimension}-dimension.csv")
