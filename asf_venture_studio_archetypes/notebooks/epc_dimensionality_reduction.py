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

# %%
# cd ~/asf_venture_studio_archetypes/

# %%
import sys
import os
import sys
import asf_core_data
import pandas as pd
from asf_core_data import load_preprocessed_epc_data
from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data import Path
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.epc_processing import *
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# define local data directory
LOCAL_DATA_DIR = "/Users/enricogavagnin/Documents/data/EPC"


# %%
# Check batch local directory
data_batches.check_for_newest_batch(
    data_path=LOCAL_DATA_DIR, check_folder="outputs", verbose=True
)

# %%
# Load preprocessed epc data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path=LOCAL_DATA_DIR,
    version="preprocessed_dedupl",
    usecols=base_epc.EPC_PREP_CLEAN_USE_FEAT_SELECTION,
    batch="newest",
    n_samples=1000000,
)

prep_epc.shape

# %% [markdown]
# ## PCA with encoded categorical features

# %% [markdown]
# ### Data cleaning
#
#

# %%
feat_drop = ["POSTCODE"]  # haven't decided how to handle it

# Drop the columns
prep_epc.drop(columns=feat_drop, inplace=True)

# Extract year of inspection date
prep_epc = extract_year_inspection(prep_epc)


# %% [markdown]
# ### Transform categorical features

# %%
# list of categorical and numerical features
cat_feat = list(
    prep_epc.columns.intersection(
        base_epc.EPC_PREP_CATEGORICAL + base_epc.EPC_PREP_ORDINAL
    )
)

# One hot encoding
encoded_features = one_hot_encoding(prep_epc, cat_feat)

# %% [markdown]
# ### Transform numerical features
#

# %%
num_feat = list(prep_epc.columns.intersection(base_epc.EPC_PREP_NUMERICAL))

# Fill missing values
prep_epc = fill_nans(prep_epc, replace_with="mean", cols=num_feat)

# Scale numeric features
scaled_features = standard_scaler(prep_epc, num_feat)

# %% [markdown]
# ### PCA analysis

# %%
processed_data = pd.concat([scaled_features, encoded_features], axis=1)
pca = pca_perform(processed_data, n_components=0.95)


# %%
pca_corr_feat = pd.DataFrame(pca.components_, columns=processed_data.columns).T

# %%

plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))


# %% [markdown]
# ### PCA exploration

# %%
# Scatter plot of the first two principal components
sns.scatterplot(x=pca_df.PCA0, y=pca_df.PCA1, hue=prep_epc["PROPERTY_TYPE"], alpha=0.4)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


# %%
from nesta_ds_utils.viz.altair.saving import save


def plot_heatmap(corr_df, width=600, height=300):
    corr_df = corr_df.reset_index()
    corr_df = corr_df.melt(id_vars="index")
    corr_df = corr_df.rename(
        columns={
            "index": "Original features",
            "variable": "Principal components",
            "value": "Correlation",
        }
    )
    heatmap = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x="Principal components:O", y="Original features:O", color="Correlation:Q"
        )
        .properties(width=width, height=height)
    )
    return heatmap


fig = plot_heatmap(corr_df, width=800, height=800)
save(fig=fig, name="pca_corr_feat", path="otuputs/figures/vegalite")

# %%
processed_data = pd.concat([scaled_features, encoded_features], axis=1)
pca = pca_perform(processed_data, n_components=2)


# %%

# %% [markdown]
# ## PCA only on scaled features
# Following the discussion with Josh, I try to perform the PCA only on numerical features, scaled and then merge back the nominal features
#

# %%
