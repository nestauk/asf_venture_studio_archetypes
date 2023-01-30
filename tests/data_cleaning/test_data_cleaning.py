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


def test_fill_nans_num():
    # Prepare test data
    df = pd.DataFrame({"col1": [1, 2, np.nan, 2]})

    # Process
    result = epc_processing.fill_nans_num(df, replace_with="mode")

    # Assert that nan is filled with mode
    assert result["col1"].tolist() == [1, 2, 2, 2]


def test_hot_one_encoding():
    # Prepare test data
    data = {
        "col1": [1, 2, 3, 4],
        "col2": ["A", "B", "C", "A"],
        "col3": [True, False, True, False],
    }
    df = pd.DataFrame(data)

    # Call the function under test
    result = epc_processing.hot_one_encoding(df, cat_feat=["col2", "col3"])

    # Assert that one-hot encoding was applied correctly
    assert result.columns.tolist() == [
        "col1",
        "col2",
        "col3",
        "col2_A",
        "col2_B",
        "col2_C",
        "col3_False",
        "col3_True",
    ]
    assert result.loc[0].tolist() == [1, "A", True, 1, 0, 0, 0, 1]
    assert result.loc[1].tolist() == [2, "B", False, 0, 1, 0, 1, 0]
    assert result.loc[2].tolist() == [3, "C", True, 0, 0, 1, 0, 1]
    assert result.loc[3].tolist() == [4, "A", False, 1, 0, 0, 1, 0]
