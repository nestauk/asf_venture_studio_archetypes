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
    epc_df: pd.DataFrame, cat_feat: Union[List[str], str] = None
) -> pd.DataFrame:
    """Performs one-hot encoding on categorical columns of a dataframe.

    Args:
        epc_df (pd.DataFrame): The input dataframe to be one-hot encoded.
        cat_feat (Union[List[str], str], optional): List of categorical column names
        or a single column name. Defaults to all categorical columns.

    Returns:
        pd.DataFrame: The one-hot encoded dataframe with non-categorical columns removed
    """

    # Get the list of categorical features
    if not cat_feat:
        cat_feat = epc_df.columns[epc_df.dtypes == object].tolist()
    elif isinstance(cat_feat, str):
        cat_feat = [cat_feat]

    # Initialize the encoded data frame
    encoded_df = pd.DataFrame()

    # One-hot encode each categorical feature
    for feat in cat_feat:
        one_hot = pd.get_dummies(epc_df[feat], prefix=feat)
        encoded_df = pd.concat([encoded_df, one_hot], axis=1)

    return encoded_df


def standard_scaler(
    epc_df: pd.DataFrame, num_feat: Union[List[str], str] = None
) -> pd.DataFrame:
    """Standardize the numerical features of a pandas DataFrame by subtracting the mean
    and scaling to unit variance.

    Args:
        epc_df (pd.DataFrame): The input DataFrame containing the features to be standardized.
        num_feat (Union[List[str], str], optional): The names of the numerical
        features to be standardized. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with the standardized numerical features, with
        non-numerical features removed
    """

    # Get the list of categorical features
    if not num_feat:
        num_feat = epc_df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(num_feat, str):
        num_feat = [num_feat]

    # Create scaler object
    scaler = StandardScaler()

    X = epc_df[num_feat].values
    X = scaler.fit_transform(X)
    return pd.DataFrame(X, columns=num_feat)


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


def averaging_construction_age_band(epc_df: pd.DataFrame):
    """Replace "CONSTRUCTION_AGE_BAND" with median year

    Args:
        epc_df (pd.DataFrame): EPC dataframe with construction age band as dataframe

    Returns:
        pd.DataFrame: EPC dataframe with construction age as numerical dtype.
    """
    epc_df["CONSTRUCTION_AGE_BAND"] = epc_df["CONSTRUCTION_AGE_BAND"].replace(
        {
            "1900-1929": 1915,
            "1930-1949": 1940,
            "1950-1966": 1958,
            "1965-1975": 1970,
            "1976-1983": 1980,
            "1983-1991": 1987,
            "1991-1998": 1995,
            "1996-2002": 1999,
            "2003-2007": 2005,
            "2007 onwards": 2007,
            "Scotland: before 1919": 1919,
            "England and Wales: before 1900": 1900,
            "unknown": np.nan,
        }
    )

    epc_df["CONSTRUCTION_AGE_BAND"] = epc_df["CONSTRUCTION_AGE_BAND"].astype("float")
    return epc_df
