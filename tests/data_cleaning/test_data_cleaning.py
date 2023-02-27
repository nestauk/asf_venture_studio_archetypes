import pytest
import sys
import pandas as pd
import numpy as np

sys.path.insert(1, "asf_venture_studio_archetypes/")
from asf_venture_studio_archetypes.utils import epc_processing


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
    epc_df = pd.DataFrame(data)

    # Call the one_hot_encoding function
    result = epc_processing.one_hot_encoding(epc_df, ["column1"])

    assert result.shape == (4, 4)
    assert result.columns.tolist() == ["column2", "column1_A", "column1_B", "column1_C"]


def test_encoder_construction_age_band():
    """Test that the encoder construction age band works as expected"""
    # test data
    data = {
        "CONSTRUCTION_AGE_BAND": [
            "1900-1929",
            "1930-1949",
            "1950-1966",
            "1965-1975",
            "1976-1983",
            "1983-1991",
            "1991-1998",
            "1996-2002",
            "2003-2007",
            "2007 onwards",
            "Scotland: before 1919",
            "England and Wales: before 1900",
            "unknown",
        ]
    }

    # Create test dataframe
    epc_df = pd.DataFrame(data)

    # Process
    result = epc_processing.encoder_construction_age_band(epc_df)

    # Assert age band replaced with median year
    assert result["CONSTRUCTION_AGE_BAND"].tolist()[0:-1] == [
        1915.0,
        1940.0,
        1958.0,
        1970.0,
        1980.0,
        1987.0,
        1995.0,
        1999.0,
        2005.0,
        2007.0,
        1919.0,
        1900.0,
    ]
    assert pd.isnull(result["CONSTRUCTION_AGE_BAND"].iloc[-1])
