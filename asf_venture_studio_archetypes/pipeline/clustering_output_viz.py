# This script contains a set of functions to analyise and interpret the clustering output

from asf_venture_studio_archetypes.utils.viz_utils import (
    plot_hist_cat_feat_cls,
    plot_scatter_joint_feat_cls,
)
import pandas as pd
from typing import Union, List
from sklearn.ensemble import RandomForestClassifier
from nesta_ds_utils.loading_saving.file_ops import make_path_if_not_exist
import numpy as np
import seaborn as sns
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


def feature_importance(
    df: pd.DataFrame,
    feat_list: Union[str, List[str]] = None,
    cluster_name: str = "cluster",
    verbose: bool = False,
    plot: bool = True,
    path: str = None,
) -> pd.DataFrame:
    """Compute the cluster feature importance. A RandomForest classifier is trained
     for each of the one-hot encoded cluster features. The importance weigths are then extract
     from the classifier and plot in separate histograms. The plots show only up to the 10th most
     important feature for visualisation purpose.

    Args:
        df (pd.DataFrame): Data frame containing the dataset.
        feat_list (Union[str, List[str]], optional): List of features used to assess the importance.
        Defaults to all df.columns, excluded the cluster feature.
        cluster_name (str, optional): Name of cluster to use for filtering, Default to "cluster".
        verbose (bool, optional): Option to printthe results as text. Defaults to False.
        plot (bool, optional): Option to show and save plot the results. Defaults to True.
        path (str, optional): Path where to save the figure. Defaults to
        "outputs/figures/clustering_interpretation/".

    Returns:
        pd.DataFrame: Data frame with the feature relative importance (cols) for different clusters (raws)
    """

    if not feat_list:
        feat_list = list(df.columns)
        feat_list.remove(cluster_name)

    if not path:
        path = "outputs/figures/clustering_interpretation/"

    make_path_if_not_exist(path)

    df_imp = pd.DataFrame(columns=feat_list)

    oh_cls = pd.get_dummies(df["cluster"])
    for c in range(max(df[cluster_name]) + 1):
        clf = RandomForestClassifier(random_state=1)
        clf.fit(df[feat_list].values, oh_cls[c].values)

        # Index sort the most important features
        sort_feat_weight_idx = np.argsort(clf.feature_importances_)[
            ::-1
        ]  # Reverse sort

        # Get the most important features names and weights
        sort_feat = np.take_along_axis(
            np.array(df[feat_list].columns.tolist()), sort_feat_weight_idx, axis=0
        )

        sort_weights = np.take_along_axis(
            np.array(clf.feature_importances_), sort_feat_weight_idx, axis=0
        )

        if verbose:
            print("\nCluster {}".format(c))
            print(
                [
                    sort_feat[i] + ": %.3f" % sort_weights[i]
                    for i in range(len(feat_list))
                ]
            )

        if plot:
            plt.figure()
            plt.title("Cluster {}".format(c))
            sns.barplot(
                x=sort_weights[: np.min([10, len(feat_list)])],
                y=sort_feat[: np.min([10, len(feat_list)])],
                palette=sns.color_palette("Blues_r", len(feat_list)),
            )
            plt.xlabel("Importance weight")
            plt.tight_layout()
            plt.savefig(path + "cls_" + str(c) + ".png")

        df_imp = pd.concat(
            [
                df_imp,
                pd.DataFrame.from_dict(
                    {sort_feat[i]: sort_weights[i] for i in range(len(feat_list))},
                    orient="index",
                ).T,
            ],
            ignore_index=True,
        )
    return df_imp


def plot_compare_feat_importance(df: pd.DataFrame, path: str = None):
    """Compare feature importance for all clusters

    Args:
        df (pd.DataFrame): Data frame with the feature relative importance (cols)
        for different clusters (raws). Output of "feature_importance"
        path (str, optional): Path where to save the figure. Defaults to
        "outputs/figures/clustering_interpretation/".
    """

    if not path:
        path = "outputs/figures/clustering_interpretation/"

    make_path_if_not_exist(path)

    plt.figure(figsize=(7, len(df.T) // 3))
    df = df.T.reset_index().melt("index", var_name="cluster", value_name="Importance")
    plt.title("Feature importance")
    sns.barplot(df, y="index", x="Importance", hue="cluster")

    # Save plot
    plt.savefig(path + "comparison_feat_importance.png")
