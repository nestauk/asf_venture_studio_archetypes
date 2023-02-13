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

# import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


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
    n_samples=10000,
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


# %% [markdown]
# ## Explore outlier removal for unprocessed features

# %%
# Load preprocessed epc data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path=LOCAL_DATA_DIR,
    version="preprocessed_dedupl",
    n_samples=5000,
    usecols=base_epc.EPC_SELECTED_FEAT,
    batch="newest",
)

# %%
sns.histplot(prep_epc.CO2_EMISS_CURR_PER_FLOOR_AREA)
prep_epc[base_epc.EPC_FEAT_NUMERICAL].describe()

# %%
new_df = remove_outliers(
    prep_epc,
    cols=[
        "TOTAL_FLOOR_AREA",
        "CO2_EMISSIONS_CURRENT",
        "CO2_EMISS_CURR_PER_FLOOR_AREA",
        "ENERGY_CONSUMPTION_CURRENT",
        "CURRENT_ENERGY_EFFICIENCY",
    ],
    remove_negative=True,
)
new_df[base_epc.EPC_FEAT_NUMERICAL].describe()

# %%
new_df

# %%
# cd ~/asf_venture_studio_archetypes/

# %%
from asf_venture_studio_archetypes.pipeline.dimensionality_reduction import *
from asf_venture_studio_archetypes.pipeline import epc_processing


# %%

processed_data = load_and_process_data()

# %%
processed_data

# %%
processed_data[processed_data.INSPECTION_DATE.isna()]

# %%
imputer = False
scaler = False
ord_encoder = False
oh_encoder = False

# Load preprocessed epc data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path=DATA_DIR,
    version="preprocessed_dedupl",
    usecols=base_epc.EPC_SELECTED_FEAT,
    batch="newest",
    n_samples=5000,  # Comment to run on full dataset (~40 min)
)

# Extract year of inspection date
if "INSPECTION_DATE" in base_epc.EPC_SELECTED_FEAT:
    prep_epc = extract_year_inspection(prep_epc)

# Outlier removal
prep_epc = remove_outliers(
    prep_epc,
    cols=[
        "TOTAL_FLOOR_AREA",
        "CO2_EMISSIONS_CURRENT",
        "CO2_EMISS_CURR_PER_FLOOR_AREA",
        "ENERGY_CONSUMPTION_CURRENT",
        "CURRENT_ENERGY_EFFICIENCY",
    ],
    remove_negative=True,
)


# if ord_encoder:   TO ADD

if imputer:
    # Fill missing values
    prep_epc = fill_nans(
        prep_epc, replace_with="mean", cols=base_epc.EPC_FEAT_NUMERICAL
    )

if scaler:
    # Standard scaling for numeric features
    prep_epc = standard_scaler(prep_epc, base_epc.EPC_FEAT_NUMERICAL)

if oh_encoder:
    # One hot encoding
    prep_epc = one_hot_encoding(prep_epc, base_epc.EPC_FEAT_NOMINAL)

prep_epc

# %% [markdown]
# ## Dev - hybrid pca
#

# %%
# cd ~/asf_venture_studio_archetypes/


# %%
from asf_venture_studio_archetypes.pipeline.dimensionality_reduction import *
from asf_venture_studio_archetypes.pipeline import epc_processing

# %%
start_time = time.time()
print("\nLoading and preprocessing EPC data.")
processed_data = load_and_process_data()
end_time = time.time()
runtime = round((end_time - start_time) / 60)
print("Loading and preprocessing the EPC data took {} minutes.\n".format(runtime))

# Perform Principal Component Analysis
start_time = time.time()
print("Performing Principal Component Analysis (PCA).")
# Selecting number of components which explain 95% of variance
pca = pca_perform(processed_data[base_epc.EPC_FEAT_NUMERICAL], n_components=0.95)
end_time = time.time()
runtime = round((end_time - start_time) / 60)
print("Principal Component Analysis took {} minutes.\n".format(runtime))

# # Saving and plotting results
# start_time = time.time()
# print("Saving and plotting results of PCA")
# # Save correlation matrix of between features and components
pca_corr_feat = pd.DataFrame(pca.components_, columns=base_epc.EPC_FEAT_NUMERICAL).T
# pca_corr_feat.to_csv("outputs/data/pca_corr_feat.csv")
# fig = plot_pca_corr_feat(pca_corr_feat)
# save(fig=fig, name="pca_corr_feat", path="outputs/figures/vegalite")
# end_time = time.time()
# runtime = round((end_time - start_time) / 60)
# print("Saving and plotting results took {} minutes.\n".format(runtime))

# %%
processed_data

# %%
np.shape(processed_data[base_epc.EPC_FEAT_NUMERICAL])[1]

pca_transformed_data = pd.DataFrame(
    pca.transform(processed_data[base_epc.EPC_FEAT_NUMERICAL].values),
    columns=["PC" + str(c) for c in range(len(pca.components_))],
)

output_data = pd.concat(
    [pca_transformed_data, processed_data[base_epc.EPC_FEAT_NOMINAL]], axis=1
)
output_data.head()

# %%
data = pd.DataFrame({"column1": ["A", "B", "C", "A"], "column2": [1, 2, 3, 4]})
one_hot_encoding(data, cols="column1")

# %%
data

# %%
prep_epc.CONSTRUCTION_AGE_BAND.unique()
{
    "1965-1975": 1970,
}
base_epc.EPC_FEAT_ORDINAL

# %%
prep_epc[base_epc.EPC_FEAT_NOMINAL]

# %%
