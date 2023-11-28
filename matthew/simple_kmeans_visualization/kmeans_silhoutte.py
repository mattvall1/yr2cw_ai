"""
    Author: Matthew Vallance 001225832
    Purpose: K-Means using the Elbow method
    Notes: https://www.w3schools.com/python/python_ml_k-means.asp
    Date: 10/11/23
"""
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

import import_data
from sklearn.cluster import KMeans


"""Visualize some data points"""
# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
x, y = import_data.get_data('lrg_rand_clusters.csv')

plt.scatter(x, y)
plt.show()


"""Visualize the inertia for different values of K"""
# Turn data into set of points to plot, list(zip())
data = list(zip(x, y))
coeffs = []

print(len(data))

# Get the highest Silhoette Coefficient to determaine how many clusters we need
for n_cluster in range(2, 25):
    kmeans = KMeans(n_clusters=n_cluster).fit(data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(data, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    coeffs.append(sil_coeff)

# Get index of highest Silhoette Coefficient (we add two to this to ger the correct number
max = max(coeffs)
n_clusters = coeffs.index(max) + 2


"""Last plot shows that 2 is a good value for K, so we retrain and visualize the result:"""
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
