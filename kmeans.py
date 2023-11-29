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
print('Data importing...')
data = import_data.get_dataframe('segmentation_data.csv')
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
data['Cluster'] = kmeans.labels_
print('-------------------- KMeans complete --------------------\n')

# Output data to CSV
filename = 'kmeans_output'
data.to_csv('outputs/' + filename + '.csv', sep=',', encoding='utf-8')
print('Outputted to ' + filename + '.csv')

