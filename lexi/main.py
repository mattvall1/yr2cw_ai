# Importing libraries
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# Loading dataset
df = pd.read_csv('99Bikers_REMOVED_ENCODED.csv', header=0)
#subset = df[['wealth_segment', 'property_valuation']]

cols_to_minmax = ['list_price','past_3_years_bike_related_purchases', 'age']
cols_one_hot = ['brand','product_line','product_size','wealth_segment','property_valuation']
# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', MinMaxScaler(feature_range=(0, 1)), cols_to_minmax),
        ('onehot', OneHotEncoder(drop='first'), cols_one_hot)
    ],
    remainder='passthrough'  # Leave other columns unchanged
)

# Fit and transform the data
df_transformed = preprocessor.fit_transform(df)
# Get the column names after scaling and one-hot encoding
columns_after_scaling = cols_to_minmax + list(preprocessor.transformers_[1][1].get_feature_names_out(cols_one_hot))
columns_after_scaling += list(df.columns[len(cols_to_minmax) + len(cols_one_hot):])
# Convert the result back to a DataFrame
df_scaled = pd.DataFrame(df_transformed, columns=columns_after_scaling)

# Scatter Plot
plt.scatter(df_scaled['past_3_years_bike_related_purchases'], df_scaled['list_price'],
            alpha=0.7, edgecolors='w')
mean_product_size = df_scaled['past_3_years_bike_related_purchases'].mean()
mean_list_price = df_scaled['list_price'].mean()
# Plot the mean values
plt.scatter(mean_product_size, mean_list_price, color='red', marker='o', s=200, label='Mean')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('',y=1.05)
plt.show()

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

plt.scatter(df.iloc[:, 2], df.iloc[:, 4], c=df['cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 2], cluster_centers[:, 4], marker='X', s=200, color='red', label='Cluster Centers')
plt.title('Mean Shift Clustering Results')
plt.xlabel('')
plt.ylabel('')
plt.show()


