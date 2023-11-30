from sklearn.cluster import AgglomerativeClustering
import import_data

# Define the data as a DataFrame
df = import_data.get_dataframe('segmentation_data.csv')

# Create an instance of Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=4)

# Fit the data
clustering.fit(df)

df['Cluster'] = clustering.labels_

filename = 'hac_output'
df.to_csv('outputs/' + filename + '.csv', sep=',', encoding='utf-8')