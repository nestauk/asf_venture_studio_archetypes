import altair as alt
import pandas as pd


def plot_pca_corr_feat(
    corr_df: pd.DataFrame, width: int = 600, height: int = 800
) -> alt.Chart:
    """Plot heatmap of correlation matrix of principal components (PCA) and original features

    Args:
        corr_df (pd.DataFrame): Correlation matrix of principal components (PCA)
        width (int, optional): Chart's width. Defaults to 600.
        height (int, optional): Chart's height . Defaults to 300.

    Returns:
        alt.Chart: heatmap of correlation matrix
    """
    corr_df = corr_df.reset_index()
    corr_df = corr_df.melt(id_vars="index")
    corr_df = corr_df.rename(
        columns={
            "index": "Original features",
            "variable": "Principal components",
            "value": "Correlation",
        }
    )
    heatmap = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x="Principal components:O", y="Original features:O", color="Correlation:Q"
        )
        .properties(width=width, height=height)
    )
    return heatmap
