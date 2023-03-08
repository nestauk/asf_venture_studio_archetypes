import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nesta_ds_utils.loading_saving.file_ops import make_path_if_not_exist


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


def plot_cat_feat_cls(df: pd.DataFrame, feat: str, path: str = None):
    """Plot multiple histogram of a categorical features filtered by cluster

    Args:
        df (pd.DataFrame): Dataframe containins the data
        feat (str): Categorical feature to plot
        path (str, optional): Path where to save the figure. Defaults to None.
    """

    if not path:
        path = "outputs/figures/clust_viz/"

    make_path_if_not_exist(path)

    sns.histplot(
        df,
        x=feat,
        hue="cluster",
        multiple="dodge",
        common_norm=False,
        stat="percent",
        shrink=0.7,
        palette=sns.color_palette(),
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + feat + ".png")


def plot_hist_cat_feat_cls(
    df: pd.DataFrame, feat: str, cluster_name: str = "cluster", path: str = None
):
    """Plot multiple histogram of a categorical feature filtered by cluster

    Args:
        df (pd.DataFrame): Dataframe containins the data
        feat (str): Categorical feature to plot
        cluster_name (str, optional): Name of cluster to use for filtering, Default to "cluster".
        path (str, optional): Path where to save the figure. Defaults to None.
    """

    if not path:
        path = "outputs/figures/clustering_viz/"

    make_path_if_not_exist(path)

    sns.histplot(
        df,
        x=feat,
        hue=cluster_name,
        multiple="dodge",
        common_norm=False,
        stat="percent",
        shrink=0.7,
        palette=sns.color_palette(),
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + feat + ".png")


def plot_scatter_joint_feat_cls(
    df: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    cluster_name: str = "cluster",
    path: str = None,
):
    """Plot joint scatter plot of two numerical features filtered by cluster

    Args:
        df (pd.DataFrame): Dataframe containins the data
        feat_x (str): Numerical feature to plot on x axis
        feat_y (str): Numerical feature to plot on y axis
        cluster_name (str, optional): Name of cluster to use for filtering, Default to "cluster".
        path (str, optional): Path where to save the figure. Defaults to None.
    """

    if not path:
        path = "outputs/figures/clustering_viz/"

    make_path_if_not_exist(path)

    sns.jointplot(
        df,
        x=feat_x,
        y=feat_y,
        hue=cluster_name,
        palette=sns.color_palette(n_colors=max(df[cluster_name]) + 1),
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + feat_x + feat_y + ".png")
