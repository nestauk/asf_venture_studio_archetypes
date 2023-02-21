import asf_core_data
from asf_core_data.getters.epc import epc_data
import time
from asf_venture_studio_archetypes.config import base_epc
from asf_venture_studio_archetypes.pipeline.epc_processing import *
from asf_venture_studio_archetypes.utils.viz_utils import plot_pca_corr_feat
from nesta_ds_utils.viz.altair.saving import save
from asf_venture_studio_archetypes.config.base_epc import DATA_DIR


def load_and_process_data():
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

    # One hot encoding
    encoded_features = one_hot_encoding(prep_epc, cat_feat)

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

    return pd.concat([scaled_features, encoded_features], axis=1)


def main():
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
    pca = pca_perform(processed_data, n_components=0.95)
    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Principal Component Analysis took {} minutes.\n".format(runtime))

    # Saving and plotting results
    start_time = time.time()
    print("Saving and plotting results of PCA")
    # Save correlation matrix of between features and components
    pca_corr_feat = pd.DataFrame(pca.components_, columns=processed_data.columns).T
    pca_corr_feat.to_csv("outputs/data/pca_corr_feat.csv")
    fig = plot_pca_corr_feat(pca_corr_feat)
    save(fig=fig, name="pca_corr_feat", path="outputs/figures/vegalite")
    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print("Saving and plotting results took {} minutes.\n".format(runtime))


if __name__ == "__main__":
    # Execute only if run as a script
    main()
