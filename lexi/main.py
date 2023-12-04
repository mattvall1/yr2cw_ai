# Importing libraries
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns


def meanshift_clustering():
    # Loading dataset
    df = pd.read_csv('99Bikers_REMOVED_ENCODED_SCALED.csv', header=0)

    # Fit the data to the Mean Shift algorithm
    bandwidth = estimate_bandwidth(df, quantile=0.025)  # 0.2122 IS THE TRAINED OPTIMAL QUANT - no scaling
    ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)  # 0.025 FOR SCALED 3, 0.02431 FOR SCALED 4
    ms.fit(df)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    silhouette_avg = silhouette_score(df, labels)
    df['cluster'] = labels
    print("number of estimated clusters : %d" % len(set(labels)))
    two_var_scatter(df, cluster_centers)
    three_dim_scatter(df)
    plot_silhouette(df, labels)


def two_var_scatter(data, cluster_centers):
    plt.scatter(data['property_valuation'], data['past_3_years_bike_related_purchases'], c=data['cluster'], cmap='Spectral')
    plt.scatter(cluster_centers[:,10], cluster_centers[:, 7], marker='o', s=50, color='red', label='Cluster Centers')
    plt.title('Mean Shift Clustering Results')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.show()


def subset_pairplot(data):
    subset_to_plot = ['list_price', 'property_valuation', 'past_3_years_bike_related_purchases', 'cluster']
    subset_df = data[subset_to_plot]
    cluster_colors = {0: '#9e0142', 1: '#5e4fa2', 2: '#fdbf6f'}
    sns.pairplot(subset_df, hue='cluster', palette=cluster_colors)
    plt.show()


def three_dim_scatter(data):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['list_price'], data['property_valuation'], data['past_3_years_bike_related_purchases'],
               c=data['cluster'], cmap='Spectral', s=50)
    # Set labels and title
    ax.set_xlabel('List Price')
    ax.set_ylabel('Property Valuation')
    ax.set_zlabel('Related Purchases in Past 3 Years')
    plt.show()


def unclustered_scatter(data):
    # Scatter Plot
    plt.scatter(data['list_price'], data['past_3_years_bike_related_purchases'], alpha=0.7, edgecolors='w')
    mean_1 = data['past_3_years_bike_related_purchases'].mean()
    mean_2 = data['list_price'].mean()
    # Plot the mean values
    plt.scatter(mean_1, mean_2, color='red', marker='o', s=50, label='Mean')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('', y=1.05)
    plt.show()


def plot_silhouette(data, labels):
    cluster_labels = np.unique(labels)
    n_clusters = len(cluster_labels)
    silhouette_vals = silhouette_samples(data, labels)

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[labels == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        colours = plt.cm.get_cmap("Spectral")(float(c) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=colours)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    avg_silhouette = silhouette_vals.mean()
    plt.axvline(x=avg_silhouette, color="red", linestyle="--", label='Average Silhouette Score')
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()

meanshift_clustering()
