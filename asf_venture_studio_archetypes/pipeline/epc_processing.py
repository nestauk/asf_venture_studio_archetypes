import pandas as pd
from typing import Union, List
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
        pd.DataFrame: The one-hot encoded dataframe.
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
        pd.DataFrame: A new DataFrame with the standardized numerical features.
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


def pca_perform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    # Transform the data using the PCA model
    X_pca = pca.transform(X)

    # Create a DataFrame from the PCA transformed data
    return pca
    return  # pd.DataFrame(X_pca, columns=['PCA%i' % i for i in range(X_pca.shape[1])])
