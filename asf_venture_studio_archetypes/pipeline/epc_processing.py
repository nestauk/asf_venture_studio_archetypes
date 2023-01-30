import pandas as pd
from typing import Union, List


def extract_year_inspection(epc_df: pd.DataFrame) -> pd.DataFrame:
    """Replace "INSPECTION_DATE" with year only

    Args:
        epc_df (pd.DataFrame): EPC dataframe with inspection date as dataframe

    Returns:
        pd.DataFrame: EPC dataframe with inspection date with year only
    """

    epc_df["INSPECTION_DATE"] = epc_df.INSPECTION_DATE.dt.year
    return epc_df


def fill_nans_num(epc_df: pd.DataFrame, replace_with: str = "mean") -> pd.DataFrame:
    """Fill nans values for numeric features with medians

    Args:
        epc_df (pd.DataFrame): EPC dataframe with original nans
        replace_with (str): specify how to replace the nans ("mean", "mode", "median"), Default: "median"

    Returns:
        pd.DataFrame: EPC dataframe with nans replaces
    """
    if replace_with == "mean":
        return epc_df.fillna(epc_df.median())
    elif replace_with == "median":
        return epc_df.fillna(epc_df.median())
    elif replace_with == "mode":
        return epc_df.apply(lambda x: x.fillna(x.mode()[0]))


def hot_one_encoding(
    epc_df: pd.DataFrame, cat_feat: Union[List[str], str] = None
) -> pd.DataFrame:
    """_summary_

    Args:
        epc_df (pd.DataFrame): _description_
        cat_feat (Union[List[str], str], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    # List of categorical features
    if not cat_feat:
        cat_feat = epc_df.columns[epc_df.dtypes == object]

    for feat in cat_feat:
        # One-hot encoding
        one_hot = pd.get_dummies(epc_df[feat]).add_prefix(feat + "_")

        # Add the one-hot encoded columns to original dataframe
        epc_df = epc_df.join(one_hot)

    return epc_df
