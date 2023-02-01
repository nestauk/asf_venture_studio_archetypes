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
# # EPC - feature selection

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

# %%
LOCAL_DATA_DIR = "/Users/enricogavagnin/Documents/data/EPC"
LOCAL_PROJ_FOLDER = "/Users/enricogavagnin/asf_venture_studio_archetypes"

# %%
# cd ~/asf_venture_studio_archetypes

# %%
# Read preprocessed data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path="S3",
    version="preprocessed_dedupl",
    usecols=None,
    batch="2021_Q4_0721",
    n_samples=5000,
)
print(prep_epc.shape)


# %%
# Read raw data
wales_epc = epc_data.load_england_wales_data(
    data_path=LOCAL_DATA_DIR, subset="Wales", batch="2021_Q2_0721"
)
wales_epc.shape

# %%
raw_feat = list(wales_epc.columns.sort_values())
clean_feat = list(prep_epc.columns.sort_values())

# %%
# Save csv file with list of features and check if they are in clean lsit

all_feat = list(set(raw_feat).union(set(clean_feat)))
all_feat.sort()
is_raw = [f in raw_feat for f in all_feat]
is_clean = [f in clean_feat for f in all_feat]
list_feat = pd.DataFrame({"feature": all_feat, "raw": is_raw, "clean": is_clean})
# list_feat.to_csv("list_feat.csv")


# %% [markdown]
# ## Read manually annotated dataframe

# %%
from asf_venture_studio_archetypes.config import base_epc

use_feat = base_epc.EPC_PREP_CLEAN_USE_FEAT_SELECTION

# %%
# Read preprocessed data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path="S3",
    version="preprocessed_dedupl",
    usecols=None,
    batch="2021_Q4_0721",
    n_samples=5000,
)
print(prep_epc.shape)

# %%
# Read preprocessed data
prep_epc = epc_data.load_preprocessed_epc_data(
    data_path="S3",
    version="preprocessed_dedupl",
    usecols=use_feat,
    n_samples=5000,
)
print(prep_epc.shape)

# %%
prep_epc

# %%
