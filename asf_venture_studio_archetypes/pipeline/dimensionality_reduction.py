import asf_core_data
from asf_core_data.getters.epc import epc_data
import time
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.epc_processing import *
from asf_venture_studio_archetypes.utils.viz_utils import plot_pca_corr_feat
from nesta_ds_utils.viz.altair.saving import save
from asf_venture_studio_archetypes.config.base_epc import DATA_DIR


def load_and_process_data(
    rem_outliers: bool = True,
    imputer: bool = True,
    scaler: bool = True,
    ord_encoder: bool = True,
    oh_encoder: bool = False,
):
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
    if rem_outliers:
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

    return prep_epc


def dim_reduction(
    numeric_only: bool = True, plot_pca_corr: bool = False, save_data: bool = False
):
    start_time = time.time()
    print("\nLoading and preprocessing EPC data.")
    processed_data = load_and_process_data()
    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Loading and preprocessing the EPC data took {} minutes.\n".format(runtime))

    # Perform Principal Component Analysis
    start_time = time.time()
    print("Performing Principal Component Analysis (PCA).")

    if numeric_only:
        # Selecting number of components which explain 95% of variance
        pca = pca_perform(
            processed_data[base_epc.EPC_FEAT_NUMERICAL], n_components=0.95
        )
        pca_transformed_data = pd.DataFrame(
            pca.transform(processed_data[base_epc.EPC_FEAT_NUMERICAL].values),
            columns=["PC" + str(c) for c in range(len(pca.components_))],
        )
        output_data = pd.concat(
            [pca_transformed_data, processed_data[base_epc.EPC_FEAT_NOMINAL]], axis=1
        )
        print(
            "Dimensionality reduction: {} ---> {} ".format(
                len(processed_data.columns), len(output_data.columns)
            )
        )

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Principal Component Analysis took {} minutes.\n".format(runtime))

    if save_data:
        output_data.to_csv("outputs/data/epc_dim_reduced_data.csv")

    if plot_pca_corr_feat:
        # Saving and plotting results
        start_time = time.time()
        print("Saving and plotting results of PCA")
        # Save correlation matrix of between features and components
        pca_corr_feat = pd.DataFrame(
            pca.components_, columns=base_epc.EPC_FEAT_NUMERICAL
        ).T
        pca_corr_feat.to_csv("outputs/data/pca_corr_feat.csv")
        fig = plot_pca_corr_feat(pca_corr_feat)
        save(fig=fig, name="pca_corr_feat", path="outputs/figures/vegalite")
        end_time = time.time()
        runtime = round((end_time - start_time) / 60)
        print("Saving and plotting results took {} minutes.\n".format(runtime))


if __name__ == "__main__":
    # Execute only if run as a script
    dim_reduction(save_data=True)
