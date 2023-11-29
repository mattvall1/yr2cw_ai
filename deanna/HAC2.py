import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('/Users/deanna/Documents/GitHub/yr2cw_ai/deanna/HAC/Mall_Customers.csv')
df.head()
df.describe()

label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df.head()

plt.figure(1, figsize = (16 ,8))
dendrogram = sch.dendrogram(sch.linkage(df, method  = "ward"))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='average')

y_hc = hc.fit_predict(df)
y_hc

df['cluster'] = pd.DataFrame(y_hc)



X = df.iloc[:, [1,2,3,4]].values
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='purple', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='orange', label ='Cluster 5')
plt.title('Clusters of Customers (Hierarchical Clustering Model)')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()

A = df.iloc[:, [2,3]].values
plt.scatter(A[y_hc==0, 0], A[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(A[y_hc==1, 0], A[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(A[y_hc==2, 0], A[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(A[y_hc==3, 0], A[y_hc==3, 1], s=100, c='purple', label ='Cluster 4')
plt.scatter(A[y_hc==4, 0], A[y_hc==4, 1], s=100, c='orange', label ='Cluster 5')
plt.title('Clusters of Customers (Hierarchical Clustering Model)')
plt.xlabel('Age')
plt.ylabel('Annual Income(k$)')
plt.show()

df.head()
df.to_csv("segmented_customers.csv", index = False)
