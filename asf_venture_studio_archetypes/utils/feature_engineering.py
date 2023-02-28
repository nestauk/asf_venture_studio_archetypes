# File: development_bank_wales/pipeline/feature_preparation/feature_engineering.py
"""Extract new features from original description features, e.g. ROOF_DESCRIPTION."""

# ---------------------------------------------------------------------------------

import pandas as pd
from asf_core_data.getters.epc import epc_data

# ---------------------------------------------------------------------------------


cats = [
    "ROOF",
    "WINDOWS",
    "WALLS",
    "FLOOR",
    "LIGHTING",
    "HOT_WATER",
    "MAINHEAT",
]


def roof_description_features(df):
    """Extract roof features from ROOF_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column ROOF_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional roof features.
    """

    df["ROOF_TYPE"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(Pitched|Flat|Roof room\(s\)|Flat|Ar oleddf|\(other premises above\)|Other premises above|\(another dwelling above\)|\(eiddo arall uwchben\))|\(annedd arall uwchben\)"
    )

    df["ROOF_TYPE"] = df["ROOF_TYPE"].replace(["Ar oleddf"], "Pitched")

    df["LOFT_INSULATION_mm"] = (
        df["ROOF_DESCRIPTION"]
        .str.extract(r"(\d{1,3})\+?\s+mm loft insulation")
        .astype(float)
    )

    df["ROOF_THERMAL_TRANSMIT"] = (
        df["ROOF_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )
    df["ROOF_INSULATION"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(no insulation|insulated at rafters|limited insulation|ceiling insulated|insulated \(assumed\))"
    )

    df["ROOF_TYPE"] = df["ROOF_TYPE"].replace(
        [
            "(other premises above)",
            "(eiddo arall uwchben)",
            "(another dwelling above)",
            "Other premises above",
            "(annedd arall uwchben)",
        ],
        "another dwelling above",
    )

    return df


def walls_description_features(df):
    """Extract walls features from WALLS_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column WALLS_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional walls features.
    """

    df["WALL_TYPE"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(Cavity wall|Sandstone|Solid brick|Sandstone or limestone|System built|Timber frame|Granite or whin|Park home wall|Waliau ceudod|Gwenithfaen|\(other premises below\)|\(another dwelling below\)|\(anheddiad arall islaw\)|\(Same dwelling below\))"
    )

    df["WALL_TYPE"] = df["WALL_TYPE"].replace(["Waliau ceudod"], "Cavity wall")
    df["WALL_TYPE"] = df["WALL_TYPE"].replace(["Gwenithfaen"], "Granite or whin")

    df["WALLS_THERMAL_TRANSMIT"] = (
        df["WALLS_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )

    df["WALLS_INSULATION"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|filled cavity|with external insulation|with internal insulation|partial insulated)"
    )

    return df


def floor_description_features(df):
    """Extract floor features from FLOOR_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column FLOOR_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional floor features.
    """

    df["FLOOR_TYPE"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(Solid|Suspended|To unheated space|Solet|To external air)"
    )

    df["FLOOR_TYPE"] = df["FLOOR_TYPE"].replace(["Solet"], "Solid")
    df["FLOOR_TYPE"] = df["FLOOR_TYPE"].replace(
        ["I ofod heb ei wresogi"], "To unheated space"
    )

    df["FLOOR_TYPE"] = df["FLOOR_TYPE"].replace(
        [
            "(other premises below)",
            "(anheddiad arall islaw)",
            "(another dwelling below)",
            "(Same dwelling below)",
        ],
        "another dwelling below",
    )

    df["FLOOR_THERMAL_TRANSMIT"] = (
        df["FLOOR_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )
    df["FLOOR_INSULATION"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|limited insulation|partial insulated|uninsulated)"
    )

    df["FLOOR_INSULATION"] = df["FLOOR_INSULATION"].replace(
        ["uninsulated"], "no insulation"
    )
    df["FLOOR_INSULATION"] = df["FLOOR_INSULATION"].replace(
        ["limited insulatio"], "partial insulated"
    )

    return df


def extract_features_from_desc(df):
    """Extract detailed features from description features such as ROOF_DESCRIPTION (available
    for each category).
    Note: Further functions, e.g. for WINDOWS, will be added in the future.

    Args:
        df (pd.DataFrame): Dataframe including description features.

    Returns:
        df (pd.DataFrame): Dataframe with new features added.
    """

    df = roof_description_features(df)
    df = walls_description_features(df)
    df = floor_description_features(df)

    return df
