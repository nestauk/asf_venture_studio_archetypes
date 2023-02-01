import pandas as pd

selected_feat = pd.read_csv(
    "asf_venture_studio_archetypes/config/epc_feature_selection.csv"
)

EPC_FEAT_SELECTION = selected_feat.feature[selected_feat.keep].tolist()

EPC_PREP_FEAT_SELECTION = selected_feat.feature[
    selected_feat.keep & selected_feat.clean
].tolist()

EPC_PREP_CLEAN_FEAT_SELECTION = selected_feat.feature[
    selected_feat.keep & selected_feat.clean & ~selected_feat.notes.notna()
].tolist()

EPC_PREP_CLEAN_USE_FEAT_SELECTION = selected_feat.feature[
    selected_feat.keep
    & selected_feat.clean
    & ~selected_feat.notes.notna()
    & selected_feat.use
].tolist()

EPC_PREP_NUMERICAL = selected_feat.feature[selected_feat.type == "numerical"].tolist()

EPC_PREP_CATEGORICAL = selected_feat.feature[
    selected_feat.type == "categorical"
].tolist()

EPC_PREP_ORDINAL = selected_feat.feature[selected_feat.type == "ordinal"].tolist()
