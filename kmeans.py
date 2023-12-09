"""
    Author: Matthew Vallance 001225832
    Purpose: K-Means algorithm
    Date: 10/11/23
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import time

# Import the data
print('Data importing and processing...')
data = pd.read_csv('inputs/99Bikers_REMOVED_ENCODED_SCALED.csv')
features = data.columns.tolist()
print('-------------------- Import complete --------------------\n')

print('KMeans training...')
start_time = time.time()
# Get the highest Silhouette Coefficient to determine how many clusters we need
coeffs = []
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(data, label, metric='euclidean')
    coeffs.append(sil_coeff)
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

end_time = time.time()
print('-------------------- Training complete | Time taken: ' + str(round(end_time - start_time, 3)) + 's --------------------\n')

# Get index of highest Silhoette Coefficient (we add two to this to ger the correct number
n_clusters = coeffs.index(max(coeffs)) + 2
print('Decided cluster length: ' + str(n_clusters) + '\n')

# Rerun KMeans after training and add the clusters to the dataframe
print('Running KMeans with decided cluster length...')
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
data['Cluster'] = pd.Series(kmeans.labels_)
print('-------------------- KMeans complete --------------------\n')

# Plotting functions
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
        color = plt.cm.get_cmap("Spectral")(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()

def silhouette_average(data, labels):
    return silhouette_score(data, labels)

def plot_matrix_graphs(df, filename):
    # Remove extra stuff we dont want to compare
    df = df.drop(['online_order', 'brand', 'product_line', 'product_size', 'gender', 'wealth_segment', 'owns_car', 'age'], axis=1)
    # Create a pair plot
    graph = sns.pairplot(df, hue='Cluster', palette='Spectral')

    plt.savefig('outputs/'+filename+'.jpg', dpi=250)
    plt.show()

plot_silhouette(data, kmeans.labels_)
plot_matrix_graphs(data, 'output_kmeans')

print(silhouette_average(data, kmeans.labels_))
