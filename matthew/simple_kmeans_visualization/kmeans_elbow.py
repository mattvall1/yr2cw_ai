"""
    Author: Matthew Vallance 001225832
    Purpose: https://www.w3schools.com/python/python_ml_k-means.asp
    Date: 10/11/23
"""
import matplotlib.pyplot as plt
import import_data
from sklearn.cluster import KMeans



"""Visualize some data points"""
# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
x, y = import_data.get_data('small_clusters2.csv')

plt.scatter(x, y)
plt.show()


"""Visualize the inertia for different values of K"""
# Turn data into set of points to plot, list(zip())
data = list(zip(x, y))
inertias = []

print(len(data))

# To find the best value of K, we draw an elbow graph,
for i in range(1, len(data)):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, len(data)), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

"""Last plot shows that 2 is a good value for K, so we retrain and visualize the result:"""
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
