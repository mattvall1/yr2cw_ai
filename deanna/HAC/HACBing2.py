import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score


def silhouette_average(data, labels):
    return silhouette_score(data, labels)


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


def dist_mat(myList):
    # creates a distance matrix
    l = myList.shape[0]
    # gives 3 columns for the two points and their distance between them to compare
    distance_matrix = np.zeros((int((l**2-l)/2), 3))
    counter = 0
    for x in range(0,(myList.shape[0])-1):
        for y in range((x+1),(myList.shape[0])):
            distance_matrix[counter] = [x, y, (np.linalg.norm(myList[x]-myList[y],axis=0))]
            counter += 1

    return(distance_matrix)


def hc(inp, k):
    # Create a model and fit it
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(inp)
    # Get labels
    label = model.labels_

    # creates a CSV file with the clustering column
    filename = 'hac_attempt2'
    df['Cluster'] = label
    df.to_csv('outputs/' + filename + '.csv', sep=',', encoding='utf-8')

    unique, counts = np.unique(label, return_counts=True)
    unique = [int(i) for i in unique]
    # how many points are in each of the clusters
    print("Cluster ID: count = " + str(dict(zip(unique, counts))))

    centroids = {}
    for x in label:
        if x == 0:
            continue
        centroids[x] = np.asarray(np.where(label == x))+1

    # Printing Centroids
    print("Cluster ID: points in cluster = ")
    for key, value in centroids.items():
        print(str(key) + ":" + str(value))

    return label

'''-------------PCA-------------'''

def plot_pca(classes_list, feature_matrix):
    # get the unique list of classes
    unique_classes_list = list(set(classes_list))

    # obtain the principle components matrix
    pca_object = PCA(n_components=2, svd_solver='full')
    pca_object.fit(feature_matrix)
    principle_components_matrix = pca_object.transform(feature_matrix)

    # Use ward as linkage metric and calculate full dendrogram
    linked = linkage(principle_components_matrix, 'ward')

    plt.figure(figsize=(10, 7))

    # Plot the dendrogram
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()
    # Determine the clusters at which to cut the dendrogram
    cluster_labels = fcluster(linked, 15, criterion='distance')

    # plot cluster_ids using the principle components as the coordinates and classes as labels
    colors = [plt.cm.jet(float(i) / max(unique_classes_list)) for i in unique_classes_list]
    for i, u in enumerate(unique_classes_list):
        xi = [p for (j,p) in enumerate(principle_components_matrix[:, 0]) if classes_list[j] == u]
        yi = [p for (j,p) in enumerate(principle_components_matrix[:, 1]) if classes_list[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(int(u)))

    plt.title(input_file.split(".")[0] + " scatter plot")
    plt.xlabel("Principle_component_1")
    plt.ylabel("Principle_component_2")
    plt.legend()
    plt.show()

    return cluster_labels

'''-------------Jaccard-------------'''


# Read categorical data directly using pandas
input_file = '99Bikers_REMOVED_ENCODED.csv'
k =3
startTime = time.time()

# Read the CSV file using pandas
df = pd.read_csv('99Bikers_REMOVED_ENCODED.csv')

data = pd.read_csv('99Bikers_REMOVED_ENCODED.csv')

ToF = ['online_order']
# One-hot encode categorical columns with 0 or 1
ordinal_encoder = OrdinalEncoder()
data[ToF] = ordinal_encoder.fit_transform(data[ToF])

# Your existing code
y = data.iloc[:, 2:]


# Removing columns with 0 variance/std
ab = np.argwhere((np.std(y,axis=0))==0)
y = np.delete(y, ab, axis=1)

# normalization
adjusted_matrix = (y - y.mean(axis=0))/(np.max(y,axis=0)-np.min(y,axis=0))

cluster_id_list = hc(adjusted_matrix, k)
# PCA
unique_cluster_id_list = list(set(cluster_id_list))
new_list = []
for cluster_id in cluster_id_list:
    new_list.append(unique_cluster_id_list.index(cluster_id))

labels = hc(data, k)
score = silhouette_average(data, labels)

print(f'Silhouette score: {score}')

endTime = time.time()
totalTime = endTime - startTime
print("Total Time: ", totalTime)
#call PCA function
plot_pca(new_list, adjusted_matrix)
plot_silhouette(data, labels)