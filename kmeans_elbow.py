"""
    Author: Matthew Vallance 001225832
    Purpose: Simple implementation of the K-Means algorithm using the elbow method
    Date: 10/11/23
"""
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



"""Visualize some data points"""
filename = 'small_clusters.csv'
x = []
y = []
with open('./inputs/'+filename, 'r') as file:
    # Read CSV file
    data = csv.reader(file)

    # Use a count to ignore headers
    count = 0
    for line in data:
        if count > 1:
            x.append(int(line[0]))
            y.append(int(line[1]))
        count += 1

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