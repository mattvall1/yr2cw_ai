"""
    Author: Matthew Vallance 001225832
    Purpose: K-Means using the Elbow method
    Notes: https://www.w3schools.com/python/python_ml_k-means.asp
    Date: 10/11/23
"""
from sklearn.metrics import silhouette_score
import import_data
from sklearn.cluster import KMeans

# Import the data
data = import_data.get_dataframe('segmentation_data.csv')

# Get the highest Silhouette Coefficient to determine how many clusters we need
coeffs = []
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(data, label, metric='euclidean')
    coeffs.append(sil_coeff)
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

# Get index of highest Silhoette Coefficient (we add two to this to ger the correct number
n_clusters = coeffs.index(max(coeffs)) + 2

# Rerun KMeans after training and add the clusters to the dataframe
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)
data['Cluster'] = kmeans.labels_

# Output data to CSV
data.to_csv('outputs/dataframe.csv', sep=',', encoding='utf-8')

