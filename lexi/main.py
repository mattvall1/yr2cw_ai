# Importing libraries
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Loading dataset
df = pd.read_csv('99Bikers_REMOVED_ENCODED.csv', header=0)

# Scaling/Encoding Data
cols_to_minmax = ['list_price', 'past_3_years_bike_related_purchases', 'age']
cols_one_hot = ['brand', 'product_line', 'product_size', 'wealth_segment', 'property_valuation']
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', MinMaxScaler(feature_range=(0, 1)), cols_to_minmax),
        ('onehot', OneHotEncoder(drop='first'), cols_one_hot)], remainder='passthrough')
# Fit and transform the data
df_transformed = preprocessor.fit_transform(df)
# Convert the result back to a DataFrame
columns_after_scaling = cols_to_minmax + list(preprocessor.transformers_[1][1].get_feature_names_out(cols_one_hot))
columns_after_scaling += list(df.columns[len(cols_to_minmax) + len(cols_one_hot):])
df_scaled = pd.DataFrame(df_transformed, columns=columns_after_scaling)

# Fit the data to the Mean Shift algorithm
bandwidth = estimate_bandwidth(df, quantile=0.2122)  # 0.2122 IS THE TRAINED OPTIMAL QUANT
ms = MeanShift(bandwidth=bandwidth)
ms.fit(df)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
try:
    silhouette_avg = silhouette_score(df, labels)
    print(f"Silhouette Score: {silhouette_avg}")
except ValueError:
    pass

df['cluster'] = labels

n_clusters = len(set(labels))
print("number of estimated clusters : %d" % n_clusters)

'''subset_to_plot = ['list_price', 'property_valuation', 'past_3_years_bike_related_purchases', 'cluster']
subset_df = df[subset_to_plot]

sns.pairplot(subset_df, hue='cluster', palette='Spectral')
plt.show()'''

'''# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(subset_df['list_price'], subset_df['property_valuation'], subset_df['past_3_years_bike_related_purchases'],
           c=df['cluster'], cmap='Spectral', s=50)
# Set labels and title
ax.set_xlabel('List Price')
ax.set_ylabel('Property Valuation')
ax.set_zlabel('Related Purchases in Past 3 Years')
plt.show()'''

'''plt.scatter(df.iloc[:, 2], df.iloc[:, 4], c=df['cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 2], cluster_centers[:, 4], marker='X', s=200, color='red', label='Cluster Centers')
plt.title('Mean Shift Clustering Results')
plt.xlabel('')
plt.ylabel('')'''

'''# Scatter Plot
plt.scatter(df_scaled['past_3_years_bike_related_purchases'], df_scaled['list_price'],
            alpha=0.7, edgecolors='w')
mean_product_size = df_scaled['past_3_years_bike_related_purchases'].mean()
mean_list_price = df_scaled['list_price'].mean()
# Plot the mean values
plt.scatter(mean_product_size, mean_list_price, color='red', marker='o', s=200, label='Mean')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('', y=1.05)
plt.show()'''