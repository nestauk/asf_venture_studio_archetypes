import pandas as pd
from typing import Union, List, Type
from sklearn.preprocessing import StandardScaler
from asf_core_data.getters.epc import epc_data
import time
import numpy as np
import pandas as pd
from asf_venture_studio_archetypes.config.base_epc import DATA_DIR


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
    df: pd.DataFrame, cat_feat: Union[List[str], str] = None
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
        cat_feat = df.columns[df.dtypes == object].tolist()
    elif isinstance(cat_feat, str):
        cat_feat = [cat_feat]

    # Initialize the encoded data frame
    encoded_df = df.drop(cat_feat, axis=1).copy()

    # One-hot encode each categorical feature
    for feat in cat_feat:
        one_hot = pd.get_dummies(df[feat], prefix=feat)
        encoded_df = pd.concat([encoded_df, one_hot], axis=1)

    return encoded_df


def remove_outliers(
    df: pd.DataFrame,
    cols: Union[List[str], str] = None,
    percentile: int = 99,
    remove_negative: bool = False,
) -> pd.DataFrame:
    """Returns DataFrame with outliers replaced with NaNs. Outliers are definied as teh values above the `percentile`,
    and negative values (if remove_negative=True).

    Args:
        df (pd.DataFrame): DataFrame to process
        feat (Union[List[str], str], optional): List of features to remove outliers from. Defaults to df.columns.
        percentile (int, optional): Percentile value to use as upper threshold for removing outliers. Defaults to 99%.
        remove_negative (bool, optional): Treat negative values as outliers. Default to False.

    Returns:
        pd.DataFrame: DataFrame with outliers replaced with NaN based on percentile theshold.
    """
    if isinstance(cols, str):
        cols = [cols]
    elif not cols:
        cols = df.columns

    for col in cols:
        threshold = np.percentile(df[~np.isnan(df[col])][col], [percentile])

        # Remove negative values
        if remove_negative:
            df[df[col] < 0] = np.nan

        # Remove outliers based on percentile thresholding
        df[df[col] > threshold[0]] = np.nan

    return df


def standard_scaler(
    df: pd.DataFrame, num_feat: Union[List[str], str] = None
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
        num_feat = df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(num_feat, str):
        num_feat = [num_feat]

    # Create scaler object
    scaler = StandardScaler()

    X = df[num_feat].values
    X = scaler.fit_transform(X)

    df[num_feat] = pd.DataFrame(X, columns=num_feat)
    return df


def encoder_construction_age_band(df: pd.DataFrame):
    """Replace "CONSTRUCTION_AGE_BAND" with median year

    Args:
        epc_df (pd.DataFrame): EPC dataframe with construction age band as dataframe

    Returns:
        pd.DataFrame: EPC dataframe with construction age as numerical dtype.
    """
    df["CONSTRUCTION_AGE_BAND"] = df["CONSTRUCTION_AGE_BAND"].replace(
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

    df["CONSTRUCTION_AGE_BAND"] = df["CONSTRUCTION_AGE_BAND"].astype("float")
    return df


def load_data(feat_list: List, n_sample: int):
    start_time = time.time()
    print("\nLoading EPC data.")
    # Load preprocessed epc data
    prep_epc = epc_data.load_preprocessed_epc_data(
        data_path=DATA_DIR,
        version="preprocessed_dedupl",
        usecols=feat_list,
        batch="newest",
        n_samples=n_sample,  # Comment to run on full dataset (~40 min)
    )
    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print(
        "Loading the EPC {} data took {} minutes.\n".format(np.shape(prep_epc), runtime)
    )

    return prep_epc


def process_data(
    prep_epc,
    epc_feat_num,
    epc_feat_cat,
    extract_year=True,
    rem_outliers: bool = True,
    imputer: bool = True,
    scaler: bool = True,
    ord_encoder: bool = True,
    oh_encoder: bool = False,
):
    start_time = time.time()
    print("\nProcessing EPC data.")

    # Extract year of inspection date
    if extract_year:
        prep_epc = extract_year_inspection(prep_epc)

    # Outlier removal
    if rem_outliers:
        rem_out_cols = list(
            set(prep_epc.columns).intersection(
                set(
                    [
                        "TOTAL_FLOOR_AREA",
                        "CO2_EMISS_CURR_PER_FLOOR_AREA",
                        "ENERGY_CONSUMPTION_CURRENT",
                        "CURRENT_ENERGY_EFFICIENCY",
                    ]
                )
            )
        )
        prep_epc = remove_outliers(
            prep_epc,
            cols=rem_out_cols,
            remove_negative=True,
        )

    if ord_encoder:
        # convert ordinal to numerical
        prep_epc = encoder_construction_age_band(prep_epc)

    if imputer:
        # Fill missing values
        prep_epc = fill_nans(prep_epc, replace_with="mean", cols=epc_feat_num)

    if scaler:
        # Standard scaling for numeric features
        prep_epc = standard_scaler(prep_epc, epc_feat_num)

    if oh_encoder and len(epc_feat_cat) > 0:
        # One hot encoding
        prep_epc = one_hot_encoding(prep_epc, epc_feat_cat)

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)
    print(
        "Processing EPC data {} took {} minutes.\n".format(np.shape(prep_epc), runtime)
    )
    return prep_epc
