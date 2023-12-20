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
inertias = []

print(len(data))

# To find the best value of K, we draw an elbow graph,
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

"""Last plot shows that 2 is a good value for K, so we retrain and visualize the result:"""
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()