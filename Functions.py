import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
import matplotlib.pyplot as plt

def fit_predict_Kmeans(X: List, number_of_clusters: int) -> List:
    """ Fits Kmeans clustering algorithm to an array.
    :return list containing the cluster labels
    """
    km = KMeans(random_state=42, n_clusters=number_of_clusters,n_init=300)
    labels = km.fit_predict(X)
    return labels

def Kmeans_searching_best_results(df: pd.DataFrame, column_combinations: List[Tuple[str,str]],
                                  nr_of_clusters_min: int, nr_of_clusters_max: int,
                                  min_sil_sample_score: float, min_sil_total_score: float,
                                  min_points_per_cluster: int) -> Dict:
    """ Loops over combinations of columns in a dataframe.
    1. Fits Kmeans clustering algorithm to every combination
    2. Filters out combinations that have a:
        * Number of datapoints in a cluster lower than param 'min_points_per_cluster'.
        * Silhoutte sample score lower than param 'min_sil_sample_score'
        * Silhouette score lower than param 'min_sil_total_score'
    3. Prints out the 10 highest values just below the parameter threshold.

    :return Dictionary containing all the good combinations with at index:
        0. Tuple of column names
        1. cluster labels as given by Kmeans
        2. Number of clusters used by Kmeans.
    """
    score_features_labels_dict = dict()
    bad_number_points = []
    bad_total_sil_scores = []
    bad_sample_sil_scores = []
    for tple in column_combinations:
       for number_of_clusters in range(nr_of_clusters_min, nr_of_clusters_max +1):
            print(number_of_clusters)
            X = df[[*tple]]
            labels = fit_predict_Kmeans(X, number_of_clusters)
            counted_labels = Counter(labels)
            least_points_in_cluster = counted_labels.most_common()[-1][1]
            if least_points_in_cluster < min_points_per_cluster:
                bad_number_points.append(least_points_in_cluster)
                continue
            sil_total_score = silhouette_score(X, labels)
            sil_sample_scores = silhouette_samples(X, labels)
            lowest_sil_sample_score = min(sil_sample_scores)
            if sil_total_score < min_sil_total_score:
                bad_total_sil_scores.append(sil_total_score)
                continue
            if lowest_sil_sample_score < min_sil_sample_score:
                bad_sample_sil_scores.append(lowest_sil_sample_score)
                continue
            print(f"Good one with silhouette_score: {sil_total_score}")
            score_features_labels_dict[sil_total_score] = (tple, labels, number_of_clusters)

    # If no scores are found that match the parameter thresholds,
    # prints the values that came closest to the parameter thresholds, to easily rerun the function.
    bad_number_points.sort()
    bad_total_sil_scores.sort()
    bad_sample_sil_scores.sort()
    print(f"Low points per cluster: {bad_number_points[-10:]}")
    print(f"Minimum total sil-score: {bad_total_sil_scores[-10:]}")
    print(bad_sample_sil_scores[-10:])
    return score_features_labels_dict


def plotting_results_of_Kmeans_searching_best_results(df: pd.DataFrame, Kmeans_best_scores: Dict):
    for key, values in Kmeans_best_scores.items():
        column_1 = values[0][0]
        column_2 = values[0][1]
        labels = values[1]
        number_of_clusters = values[2]
        print(f"Features:{column_1}, {column_2}, Score:{key}, Number of clusters:{number_of_clusters}")
        sns.scatterplot(data=df, x= column_1, y=column_2, hue=labels)
        plt.title(f"Silhouette score for {number_of_clusters} clusters: {key}")
        plt.xlabel(f"{column_1}")
        plt.ylabel(f"{column_2}")
        plt.show()






# Was short on time to put this (for plotting silhouette score comparison) into a function.
# for feature1,feature2 in combinations:
#     X = df_scaled[[feature1, feature2]].values
#     range_n_clusters = [2, 3, 4, 5, 6]
#     silhouette_avg_n_clusters = []
#     for n_clusters in range_n_clusters:
#         # Create a subplot with 1 row and 2 columns
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         fig.set_size_inches(18, 7)
#
#         # The 1st subplot is the silhouette plot
#         # The silhouette coefficient can range from -1, 1 but in this example all
#         # lie within [-0.1, 1]
#         ax1.set_xlim([-0.4, 1])
#         # The (n_clusters+1)*10 is for inserting blank space between silhouette
#         # plots of individual clusters, to demarcate them clearly.
#         ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#         # Initialize the clusterer with n_clusters value and a random generator
#         # seed of 10 for reproducibility.
#         clusterer = KMeans(n_clusters=n_clusters, random_state=42,n_init=400,max_iter=500)
#         cluster_labels = clusterer.fit_predict(X)
#
#         # The silhouette_score gives the average value for all the samples.
#         # This gives a perspective into the density and separation of the formed
#         # clusters
#         silhouette_avg = silhouette_score(X, cluster_labels)
#         print(f"For {n_clusters} clusters:  "
#               "The average silhouette_score is :", silhouette_avg)
#
#         silhouette_avg_n_clusters.append(silhouette_avg)
#         # Compute the silhouette scores for each sample
#         sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#         y_lower = 10
#         for i in range(n_clusters):
#             # Aggregate the silhouette scores for samples belonging to
#             # cluster i, and sort them
#             ith_cluster_silhouette_values = \
#                 sample_silhouette_values[cluster_labels == i]
#
#             ith_cluster_silhouette_values.sort() #Per cluster, aantal punten + score.
#
#
#             size_cluster_i = ith_cluster_silhouette_values.shape[0]
#             y_upper = y_lower + size_cluster_i
#
#             color = cm.nipy_spectral(float(i) / n_clusters)
#             ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                               0, ith_cluster_silhouette_values,
#                               facecolor=color, edgecolor=color, alpha=0.7)
#
#             # Label the silhouette plots with their cluster numbers at the middle
#             ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#             # Compute the new y_lower for next plot
#             y_lower = y_upper + 10  # 10 for the 0 samples
#
#         ax1.set_title("The silhouette plot for the various clusters.")
#         ax1.set_xlabel("The silhouette coefficient values")
#         ax1.set_ylabel("Cluster label")
#
#         # The vertical line for average silhouette score of all the values
#         ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#         ax1.set_yticks([])  # Clear the yaxis labels / ticks
#         ax1.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#         # 2nd Plot showing the actual clusters formed
#         colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#         ax2.scatter(X[:, 0], X[:, 1], marker='.', s=45, lw=0, alpha=0.7,
#                     c=colors, edgecolor='k')
#
#         # Labeling the clusters
#         centers = clusterer.cluster_centers_
#         # Draw white circles at cluster centers
#         ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                     c="white", alpha=1, s=200, edgecolor='k')
#
#         for i, c in enumerate(centers):
#             ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                         s=50, edgecolor='k')
#
#         ax2.set_title("The visualization of the clustered data.")
#         ax2.set_xlabel(f"{feature1}")
#         ax2.set_ylabel(f"{feature2}")
#
#         plt.suptitle((f"Silhouette score for KMeans clustering on data "
#                       f"with {n_clusters} clusters = {silhouette_avg} "),
#                      fontsize=14, fontweight='bold')
#     plt.show()
#
#
# style.use("fivethirtyeight")
# plt.plot(range_n_clusters, silhouette_avg_n_clusters)
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("silhouette score")
# plt.show()