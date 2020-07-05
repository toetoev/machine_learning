import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
from processing import preprocessing, pca, feature_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage


def kmeans(df, y, n_clusters=2, features=0, show_elbow=False):
    df = preprocessing(data=df, y=df[y], perform_scale=True)
    if features == 0:
        df = feature_selection(df=df, target=df[y], show_process=False)
    elif features == 1:
        df = pca(df.loc[:, df.columns.difference([y])],
                 df[y], 0.8, show_result=False)
    x = df.loc[:, df.columns.difference([y])]

    # Applying kmeans to the dataset / Creating the kmeans classifier
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(x.values)

    # 2D plot
    colors = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

    plt.subplot(2, 2, 1)
    for i in np.unique(clusters):
        plt.scatter(x.iloc[clusters == i, 0], x.iloc[clusters == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=100, c='lightskyblue', label='Centroids')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    plt.subplot(2, 2, 2)
    for i in np.unique(df[y].values):
        plt.scatter(x.iloc[df[y].values == i, 0], x.iloc[df[y].values == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Ground Truth Classification')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()

    # Part 2: Find the optimum number of clusters for k-means
    if show_elbow:
        wcss = []
        # Trying kmeans for k=1 to k=10
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++')
            kmeans.fit(x)
            wcss.append(kmeans.inertia_)

        # Plotting the results onto a line graph, allowing us to observe 'The elbow'
        plt.plot(range(1, 11), wcss)
        plt.title('The elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')  # within cluster sum of squares
        plt.show()


def dbscan(df, y, eps=0.5, min_samples=5, features=0):
    df = preprocessing(data=df, y=df[y], perform_scale=True)
    if features == 0:
        df = feature_selection(df=df, target=df[y], show_process=False)
    elif features == 1:
        df = pca(df.loc[:, df.columns.difference([y])],
                 df[y], 0.8, show_result=False)
    x = df.loc[:, df.columns != y]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(x)

    colors = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

    plt.subplot(2, 2, 1)
    for i in np.unique(clusters):
        label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
        plt.scatter(x.iloc[clusters == i, 0], x.iloc[clusters == i, 1],
                    color=colors[i % 3], label=label)

    plt.legend()
    plt.title('DBSCAN Clustering')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    plt.subplots_adjust(wspace=0.4)

    plt.subplot(2, 2, 2)
    for i in np.unique(df[y].values):
        plt.scatter(x.iloc[df[y].values == i, 0], x.iloc[df[y].values == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Ground Truth Classification')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()


def hierarchical(df, y, n_clusters=2, scaling=True, features=0, show_dendrogram=False):
    df = preprocessing(data=df, y=df[y], perform_scale=scaling)
    if features == 0:
        df = feature_selection(df=df, target=df[y], show_process=False)
    elif features == 1:
        df = pca(df.loc[:, df.columns.difference([y])],
                 df[y], 2, show_result=False)
    print(df)
    x = df.loc[:, df.columns.difference([y])]
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel(y)
    plt.ylabel('distance')

    if show_dendrogram:
        dendrogram(
            linkage(x, 'ward'),  # generate the linkage matrix
            leaf_font_size=8      # font size for the x axis labels
        )

        plt.axhline(y=8)
        plt.show()

    clusters = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
    clusters.fit(x)

    print(clusters.labels_)

    colors = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

    plt.subplot(2, 2, 1)
    for i in np.unique(clusters.labels_):
        plt.scatter(x.iloc[clusters.labels_ == i, 0], x.iloc[clusters.labels_ == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Hierarchical Clustering')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    plt.subplot(2, 2, 2)
    for i in np.unique(df[y].values):
        plt.scatter(x.iloc[df[y].values == i, 0], x.iloc[df[y].values == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Ground Truth Classification')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()
