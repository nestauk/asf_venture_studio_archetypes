# This script contains a set of functions to analyise and interpret the clustering output

from asf_venture_studio_archetypes.utils.viz_utils import (
    plot_hist_cat_feat_cls,
    plot_scatter_joint_feat_cls,
)
import pandas as pd
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt


def plot_dist_cls(
    df: pd.DataFrame,
    hist_feat: Union[str, List[str]] = None,
    scatter_feat: Union[str, List[str]] = None,
    path: str = None,
    cluster_name: str = "cluster",
):
    """Plot all the distribution plots either as histogram (if categorical), or
    as joint scatter plot (if numerical), with cluster label as filter. Unless specified otherwise.
    all the categorical features are plotted as histogram and all the numerical as joint scatter plots

    WARNING: The joint scatter plots are currently pairing the list of numerical features in
    sequential order, leaving the last behind if the list has a odd length. This should be improved.

    Args:
        df (pd.DataFrame): _description_
        hist_feat (Union[str, List[str]], optional): List of features to plot as histograms. Defaults to None.
        scatter_feat (Union[str, List[str]], optional): List of features to plot as joint scatter plots. Defaults to None.
        path (str, optional): Path where to save the figure. Defaults to None.
        cluster_name (str, optional): Name of cluster to use for filtering, Default to "cluster".
    """
    if not hist_feat:
        hist_feat = list(
            set(df.columns) - set(df.select_dtypes(include=np.number).columns)
        )

    if not scatter_feat:
        scatter_feat = list(df.select_dtypes(include=np.number).columns)
        scatter_feat.remove(cluster_name)

    if not path:
        path = "outputs/figures/clustering_viz/"

    for c_feat in hist_feat:
        plt.figure()
        plot_hist_cat_feat_cls(df, feat=c_feat, path=path, cluster_name=cluster_name)

    for feat_idx in range(0, len(scatter_feat) - 1, 2):
        feat_x = scatter_feat[feat_idx]
        feat_y = scatter_feat[feat_idx + 1]
        plt.figure()
        plot_scatter_joint_feat_cls(
            df, feat_x=feat_x, feat_y=feat_y, path=path, cluster_name=cluster_name
        )
