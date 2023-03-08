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
# # K-means
# ### Use example

# %%
# cd ~/asf_venture_studio_archetypes

# %%
from asf_venture_studio_archetypes.pipeline.k_means import *
from asf_venture_studio_archetypes.utils.epc_processing import *
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.clustering_output_viz import *

# Load features sets to use
feat_list_num = base_epc.EPC_FEAT_NUM_KMEANS_NOSCORES
feat_list_cat = base_epc.EPC_FEAT_CAT_KMEANS_NOSCORES

feat_list_num = feat_list_num.copy()
feat_list_cat = feat_list_cat.copy()
feat_list = feat_list_num + feat_list_cat

# Sample size
n_sample = 10000

# Loading and processing
load_df = load_data(feat_list, n_sample)
pro_df = process_data(load_df, feat_list_num, feat_list_cat, oh_encoder=True)

# %% [markdown]
# # Silhouette Analysis

# %%
KMeans_silhouette_analysis(pro_df, range(2, 10), reps=5)

# %% [markdown]
# # Clustering interpretation

# %% [markdown]
# Perform K-means with 5 clusters (chosen as example) and minimum cluster size 400.

# %%
load_df["cluster"] = KMeans_apply(
    pro_df.copy(), n_clusters=5, size_min=400, n_init=10
).cluster
pro_df["cluster"] = load_df["cluster"]

# %% [markdown]
# Plot features relative importance

# %%
df_imp = feature_importance(pro_df.copy())
plot_compare_feat_importance(df_imp.copy())

# %% [markdown]
# Plot marginal distributions for filter by cluster labelling

# %%
plot_dist_cls(load_df)
