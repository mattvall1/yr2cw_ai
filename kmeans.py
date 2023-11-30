"""
    Author: Matthew Vallance 001225832
    Purpose: K-Means using the Elbow method
    Notes: https://www.w3schools.com/python/python_ml_k-means.asp
    Date: 10/11/23
"""
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder
import import_data
from sklearn.cluster import KMeans
import time

# Import the data
print('Data importing and processing...')
data = pd.read_csv('inputs/shopping_trends.csv')
original_data = pd.read_csv('inputs/shopping_trends.csv')
headers = import_data.get_dataframe('shopping_trends.csv')

# TODO: Feature selection?

ordinal_encoder = OrdinalEncoder()
data[headers] = ordinal_encoder.fit_transform(data[headers])

print('-------------------- Import complete --------------------\n')

print('KMeans training...')
# Get the highest Silhouette Coefficient to determine how many clusters we need
coeffs = []
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(data, label, metric='euclidean')
    coeffs.append(sil_coeff)
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
print('-------------------- Training complete --------------------\n')

# Get index of highest Silhoette Coefficient (we add two to this to ger the correct number
n_clusters = coeffs.index(max(coeffs)) + 2
print('Decided cluster length: ' + str(n_clusters) + '\n')

# Rerun KMeans after training and add the clusters to the dataframe
print('Running KMeans with decided cluster length...')
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)

original_data['Cluster'] = kmeans.labels_
print('-------------------- KMeans complete --------------------\n')

# Output data to CSV
filename = 'kmeans_output'
original_data.to_csv('outputs/' + filename + '.csv', sep=',', encoding='utf-8', index=False)
print('Outputted to ' + filename + '.csv')

