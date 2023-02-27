import time
from typing import List
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.utils.epc_processing import *
from sklearn.decomposition import PCA
from asf_venture_studio_archetypes.utils.viz_utils import plot_pca_corr_feat
from nesta_ds_utils.viz.altair.saving import save


def PCA_numeric_only(
    feat_list_num: List,
    feat_list_cat: List,
    n_sample: int = 10000,
    plot_pca_corr: bool = False,
    save_data: bool = False,
):
    """Pipeline that load, process and reduce dimensionality by PCA on the
    numerical features only.

    Args:
        feat_list_num (List): List of numerical features to load
        feat_list_cat (List): List of categorical features to load
        n_sample (int): Number of sample to load. Defaults tp 100000
        plot_pca_corr (bool, optional): Plot correlation plot of PCA components. Defaults to False.
        save_data (bool, optional): Save output data. Defaults to False.
    """

    feat_list = feat_list_num + feat_list_cat

    # Load data
    processed_data = load_data(feat_list, n_sample)

    # Process data
    processed_data = process_data(
        processed_data, feat_list_num, feat_list_cat, oh_encoder=False
    )

    # Perform Principal Component Analysis
    start_time = time.time()
    print("Performing Principal Component Analysis (PCA).")

    # Selecting number of components which explain 95% of variance
    X = processed_data[feat_list_num].values
    pca = PCA(n_components=0.95).fit(X)

    pca_transformed_data = pd.DataFrame(
        pca.transform(X),
        columns=["PC" + str(c) for c in range(len(pca.components_))],
    )

    output_data = pd.concat([pca_transformed_data, processed_data], axis=1)

    print(
        "Dimensionality reduction: {} ---> {} ".format(
            len(feat_list_num), len(pca.components_)
        )
    )

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Principal Component Analysis took {} minutes.\n".format(runtime))

    if save_data:
        output_data.to_csv("outputs/data/epc_dim_reduced_data.csv")

    if plot_pca_corr:
        # Saving and plotting results
        start_time = time.time()
        print("Saving and plotting results of PCA")
        # Save correlation matrix of between features and components
        pca_corr_feat = pd.DataFrame(pca.components_, columns=feat_list_num).T
        pca_corr_feat.to_csv("outputs/data/pca_corr_feat.csv")
        fig = plot_pca_corr_feat(pca_corr_feat)
        save(fig=fig, name="pca_corr_feat", path="outputs/figures/vegalite")
        end_time = time.time()
        runtime = round((end_time - start_time) / 60)
        print("Saving and plotting results took {} minutes.\n".format(runtime))


if __name__ == "__main__":
    # Execute only if run as a script
    PCA_numeric_only(
        feat_list_num=base_epc.EPC_FEAT_NUM_KMEANS,
        feat_list_cat=base_epc.EPC_FEAT_CAT_KMEANS,
    )
