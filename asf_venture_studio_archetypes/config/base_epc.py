import pandas as pd

# Data directory
# You can set this to your local data directory or "S3"
# DATA_DIR = "S3"
DATA_DIR = "/Users/enricogavagnin/Documents/data/EPC"


# Feature selection
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

EPC_FEAT_NUMERICAL = [
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "CURRENT_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    #'ENERGY_CONSUMPTION_POTENTIAL',
    #'EXTENSION_COUNT',
    #'HEATING_COST_CURRENT',
    #'HOT_WATER_COST_CURRENT',
    "HOT_WATER_ENERGY_EFF_SCORE",
    "INSPECTION_DATE",
    #'LIGHTING_COST_CURRENT',
    "LIGHTING_ENERGY_EFF_SCORE",
    "LOW_ENERGY_LIGHTING",
    "MAINHEATC_ENERGY_EFF_SCORE",
    "MAINHEAT_ENERGY_EFF_SCORE",
    "MULTI_GLAZE_PROPORTION",
    "NUMBER_HABITABLE_ROOMS",
    #'NUMBER_HEATED_ROOMS',
    #'NUMBER_OPEN_FIREPLACES',
    "ROOF_ENERGY_EFF_SCORE",
    "TOTAL_FLOOR_AREA",
    "WALLS_ENERGY_EFF_SCORE",
    "WINDOWS_ENERGY_EFF_SCORE",
]

EPC_FEAT_NOMINAL = [
    "BUILT_FORM",
    "ENERGY_TARIFF",
    #'FLOOR_DESCRIPTION',
    "GLAZED_TYPE",
    "HEATING_FUEL",
    "HEATING_SYSTEM",
    #'HOTWATER_DESCRIPTION',
    "HP_INSTALLED",
    "HP_TYPE",
    #'LIGHTING_DESCRIPTION',
    #'MAINHEATCONT_DESCRIPTION',
    #'MAINHEAT_DESCRIPTION',
    "MAINS_GAS_FLAG",
    #'POSTCODE', #havn't decided how to handle it yet
    "PROPERTY_TYPE",
    #'ROOF_DESCRIPTION',
    #'SECONDHEAT_DESCRIPTION',
    "SOLAR_WATER_HEATING_FLAG",
    "TENURE",
    "TRANSACTION_TYPE",
    #'WALLS_DESCRIPTION',
    #'WINDOWS_DESCRIPTION'
]

EPC_FEAT_ORDINAL = ["CONSTRUCTION_AGE_BAND", "CURRENT_ENERGY_RATING"]

EPC_SELECTED_FEAT = EPC_FEAT_NUMERICAL + EPC_FEAT_NOMINAL + EPC_FEAT_ORDINAL
