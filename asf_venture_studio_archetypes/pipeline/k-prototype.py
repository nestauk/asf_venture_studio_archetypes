import pandas as pd
from asf_core_data import load_preprocessed_epc_data
from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data import Path
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.epc_processing import *
from kmodes.kprototypes import KPrototypes
from plotnine import *
import plotnine

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

# Further data cleaning
feat_drop = ["POSTCODE"]  # haven't decided how to handle it yet
prep_epc.drop(columns=feat_drop, inplace=True)

# Extract year of inspection date
prep_epc = extract_year_inspection(prep_epc)

# convert ordinal to numerical
prep_epc = encoder_construction_age_band(prep_epc)

# Transform categorical features
cat_feat = list(prep_epc.columns.intersection(base_epc.EPC_PREP_CATEGORICAL))

# # One hot encoding
# encoded_features = one_hot_encoding(prep_epc, cat_feat)

# Transform numerical features
num_feat = list(
    prep_epc.columns.intersection(
        base_epc.EPC_PREP_NUMERICAL + base_epc.EPC_PREP_ORDINAL
    )
)

# Fill missing values
prep_epc = fill_nans(prep_epc, replace_with="mean", cols=num_feat)

# Scale numeric features
scaled_features = standard_scaler(prep_epc, num_feat)
categorical_features = prep_epc[cat_feat]

prep_epc = pd.concat([scaled_features, categorical_features], axis=1)

# Get the position of categorical columns
df = prep_epc


catColumnsPos = [
    df.columns.get_loc(col)
    for col in list(df.select_dtypes(["object", "bool"]).columns)
]
print(
    "Categorical columns           : {}".format(
        list(df.select_dtypes("object", "bool").columns)
    )
)
print("Categorical columns pocatsition  : {}".format(catColumnsPos))

# Convert dataframe to matrix
dfMatrix = df.to_numpy()

# # Choose optimal K using Elbow method
# cost = []
# for cluster in range(1, 20):
#     try:
#         kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
#         kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
#         cost.append(kprototype.cost_)
#         print('Cluster initiation: {}'.format(cluster))
#     except:
#         break
# # Converting the results into a dataframe and plotting them
# df_cost = pd.DataFrame({'Cluster':range(1, len(cost)+1), 'Cost':cost})
# # Data viz
# plotnine.options.figure_size = (8, 4.8)
# (
#     ggplot(data = df_cost)+
#     geom_line(aes(x = 'Cluster',
#                   y = 'Cost'))+
#     geom_point(aes(x = 'Cluster',
#                    y = 'Cost'))+
#     geom_label(aes(x = 'Cluster',
#                    y = 'Cost',
#                    label = 'Cluster'),
#                size = 10,
#                nudge_y = 1000) +
#     labs(title = 'Optimal number of cluster with Elbow Method')+
#     xlab('Number of Clusters k')+
#     ylab('Cost')+
#     theme_minimal()
# )
# # Choosing cluster 12

# Fit the cluster
kprototype = KPrototypes(n_jobs=-1, n_clusters=12, init="Huang", random_state=0)
kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

# Cluster centorid
kprototype.cluster_centroids_
# Check the iteration of the clusters created
kprototype.n_iter_
# Check the cost of the clusters created
kprototype.cost_

# Add the cluster to the dataframe
df["Cluster Labels"] = kprototype.labels_
