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
# cd ~/asf_venture_studio_archetypes

# %%
import sys
import os
import asf_core_data
import pandas as pd

from asf_core_data import load_preprocessed_epc_data
from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.pipeline.preprocessing import preprocess_epc_data
from asf_core_data.utils.visualisation import easy_plotting
from asf_core_data.getters import data_getters
from asf_core_data.config import base_config
from asf_core_data.pipeline.data_joining import merge_install_dates
from asf_core_data import Path
from asf_venture_studio_archetypes.config import base_epc

# %%
LOCAL_DATA_DIR = "/Users/enricogavagnin/Documents/data/EPC"
LOCAL_PROJ_FOLDER = "/Users/enricogavagnin/asf_venture_studio_archetypes"


# %%

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
    n_samples=100000,
)

prep_epc.shape

# %%

# %% [markdown]
# ## PCA with encoded categorical features

# %% [markdown]
# ## Data cleaning

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# %%
feat_drop = [
    "POSTCODE",  # haven't decided how to handle it
    "MAIN_HEATING_CONTROLS",  # needs further cleaning
    "FLOOR_LEVEL",  # too many nans
    "GLAZED_AREA",  # too many nans
    "FLOOR_ENERGY_EFF_SCORE",
]  # too many nans

# Drop the columns
prep_epc.drop(columns=feat_drop, inplace=True)

# %%
# Extract year form INSPECTION_DATE
prep_epc["INSPECTION_DATE"] = prep_epc.INSPECTION_DATE.dt.year


# %%
prep_epc[prep_epc.columns[prep_epc.dtypes == object]].describe()

# %% [markdown]
# ## Fill nans

# %%
# Fill nans for numeric features

# Compute the medians of each column
medians = prep_epc.mode(numeric_only=True)

# Use the medians to fill the missing values
prep_epc.fillna(medians, inplace=True)


# %%
import numpy as np

df = pd.DataFrame({"col1": [1, 2, np.nan, 2]})
# Fill nans for numeric features

df.apply(lambda x: x.fillna(x.mode()[0]))

# %% [markdown]
# ## One-hot encode categorical features

# %%
# list of non numerical feat
cat_feat = prep_epc.columns[prep_epc.dtypes == object]
cat_feat

# %%
for feat in cat_feat:

    # One-hot encode the 'category' column
    one_hot = pd.get_dummies(prep_epc[feat]).add_prefix(feat + "_")

    # Add the one-hot encoded columns to the DataFrame
    prep_epc = prep_epc.join(one_hot)


# %%
# Create dataframe with numerical features
prep_epc_num = prep_epc.drop(cat_feat, axis=1)


# %% [markdown]
# ### PCA analysis

# %%
# Select the features for PCA
X = prep_epc_num.values

# Create a PCA object
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data using the PCA model
X_pca = pca.transform(X)

# Create a DataFrame from the PCA transformed data
pca_df = pd.DataFrame(X_pca, columns=["PCA%i" % i for i in range(X_pca.shape[1])])

# %%
pca.explained_variance_ratio_

# %%

# Scatter plot of the first two principal components
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=prep_epc["PROPERTY_TYPE"], alpha=0.4)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


# %%
corr_df = pd.DataFrame(pca.components_, columns=prep_epc_num.columns)
corr_df = corr_df.T
print(corr_df.iloc[:10, :])

# %%
sns.scatterplot(
    data=prep_epc,
    x="ENERGY_CONSUMPTION_CURRENT",
    y="TOTAL_FLOOR_AREA",
    hue="PROPERTY_TYPE",
)

# %%


# %%
