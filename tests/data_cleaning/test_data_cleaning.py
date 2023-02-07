import pytest
import sys
import pandas as pd
import numpy as np

sys.path.insert(1, "asf_venture_studio_archetypes/")
from asf_venture_studio_archetypes.pipeline import epc_processing


def test_extract_year_inspection():
    """Test that the INSPECTION_DATE column of the processed dataframe
    contains only the year extracted
    """

    df = pd.DataFrame(
        {
            "INSPECTION_DATE": [
                np.datetime64("2021-01-01"),
                np.datetime64("2022-02-02"),
                np.datetime64("2023-03-03"),
            ]
        }
    )

    pd.testing.assert_frame_equal(
        epc_processing.extract_year_inspection(df),
        pd.DataFrame({"INSPECTION_DATE": [2021, 2022, 2023]}),
    )


def test_fill_nans():
    """Test that fill nans with "mode" option return the expected value"""
    # Prepare test data
    df = pd.DataFrame({"col1": [1, 2, np.nan, 2]})

    # Process
    result = epc_processing.fill_nans(df, replace_with="mode")

    # Assert that nan is filled with mode
    assert result["col1"].tolist() == [1, 2, 2, 2]


def test_one_hot_encoding():
    """Test that the one hot encoding works as expected"""
    # Test data
    data = {"column1": ["A", "B", "C", "A"], "column2": [1, 2, 3, 4]}

    # Create test dataframe
    epc_df_test = pd.DataFrame(data)

    # Call the one_hot_encoding function
    result = epc_processing.one_hot_encoding(epc_df_test, ["column1"])

    assert result.shape == (4, 4)
    assert result.columns.tolist() == [
        "column2",
        "column1_A",
        "column1_B",
        "column1_C",
    ]


def test_remove_outliers():
    """Test remove outlier  funciton"""
    # Setup test data
    df = pd.DataFrame({"A": [1, 2, 3, 4, 102], "B": [2, 3, 4, 5, 6]})

    # Test remove_outliers function with a specified column
    df_out = epc_processing.remove_outliers(df, cols="A")
    assert np.isnan(df.iloc[4, 0])
