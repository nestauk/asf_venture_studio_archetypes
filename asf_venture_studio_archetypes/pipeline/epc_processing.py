import pandas as pd
from typing import Union, List, Type
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def extract_year_inspection(epc_df: pd.DataFrame) -> pd.DataFrame:
    """Replace "INSPECTION_DATE" with year only

    Args:
        epc_df (pd.DataFrame): EPC dataframe with inspection date as dataframe

    Returns:
        pd.DataFrame: EPC dataframe with inspection date with year only
    """

    epc_df["INSPECTION_DATE"] = epc_df.INSPECTION_DATE.dt.year
    return epc_df


def fill_nans(
    epc_df: pd.DataFrame, replace_with: str = "mean", cols: Union[List[str], str] = None
) -> pd.DataFrame:
    """Fill nans values for numeric features with medians

    Args:
        epc_df (pd.DataFrame): EPC dataframe with original nans
        replace_with (str): specify how to replace the nans ("mean", "mode", "median"). Default: "median"
        cols (Union[List[str], str]): specify which column to apply the fill nan function,
        Default: all numeric columns

    Returns:
        pd.DataFrame: EPC dataframe with nans replaces
    """
    if cols is None:
        cols = epc_df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]

    if replace_with == "mean":
        epc_df[cols] = epc_df[cols].fillna(epc_df[cols].mean())
    elif replace_with == "median":
        epc_df[cols] = epc_df[cols].fillna(epc_df[cols].median())
    elif replace_with == "mode":
        epc_df[cols] = epc_df[cols].apply(lambda x: x.fillna(x.mode()[0]))

    return epc_df


def one_hot_encoding(
    epc_df: pd.DataFrame, cols: Union[List[str], str] = None
) -> pd.DataFrame:
    """Performs one-hot encoding on categorical columns of a dataframe.

    Args:
        epc_df (pd.DataFrame): The input dataframe to be one-hot encoded.
        cols (Union[List[str], str], optional): List of nominal column names
        or a single column name. Defaults to all categorical columns.

    Returns:
        pd.DataFrame: Original DataFrame with the one-hot encoded nominal features
    """

    # Get the list of categorical features
    if not cols:
        cols = epc_df.columns[epc_df.dtypes == object].tolist()
    elif isinstance(cols, str):
        cols = [cols]

    # One-hot encode each categorical feature
    for feat in cols:
        one_hot = pd.get_dummies(epc_df[feat], prefix=feat)
        epc_df = pd.concat([epc_df, one_hot], axis=1).drop([feat], axis=1)
    return epc_df


def standard_scaler(
    epc_df: pd.DataFrame, cols: Union[List[str], str] = None
) -> pd.DataFrame:
    """Standardize the numerical features of a pandas DataFrame by subtracting the mean
    and scaling to unit variance.

    Args:
        epc_df (pd.DataFrame): The input DataFrame containing the features to be standardized.
        cols (Union[List[str], str], optional): The names of the numerical
        features to be standardized. Defaults to None.

    Returns:
        pd.DataFrame: Original DataFrame with the standardized numerical features
    """

    # Get the list of categorical features
    if not cols:
        cols = epc_df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]

    # Create scaler object
    scaler = StandardScaler()

    X = epc_df[cols].values
    X = scaler.fit_transform(X)
    epc_df[cols] = pd.DataFrame(X, columns=cols)
    return epc_df


def remove_outliers(
    df: pd.DataFrame, cols: Union[List[str], str] = None, percentile: int = 99
) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): DataFrame to process
        feat (Union[List[str], str], optional): List of features to remove outliers from. Defaults to df.columns.
        percentile (int, optional): Percentile value to use as upper threshold for removing outliers. Defaults to 99%.

    Returns:
        pd.DataFrame: DataFrame with outliers removes based on percentile theshold.
    """
    if isinstance(cols, str):
        cols = [cols]
    elif not cols:
        cols = df.columns

    for col in cols:
        threshold = np.percentile(df[~np.isnan(df[col])][col], [percentile])

        # Remove outliers
        df = df[df[col] <= threshold[0]]

    return df


def pca_perform(df: pd.DataFrame, **kwargs) -> Type[PCA]:
    """Perform Principal Component Analysis (PCA) on dataframe

    Args:
        df (pd.DataFrame): Dataframe to analyise
        n_components (int, optional): Number of component for PCA. Defaults to 2.

    Returns:
        pd.DataFrame: Dataframe with PCA transformed data
    """

    X = df.values

    # Create a PCA object
    pca = PCA(**kwargs)

    # Fit the PCA model to the data
    pca.fit(X)

    return pca
