# Importing libraries
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


def meanshift_clustering():
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
    silhouette_avg = silhouette_score(df, labels)
    df['cluster'] = labels
    print("number of estimated clusters : %d" % len(set(labels)))
    three_dim_scatter(df)


def two_var_scatter(data, cluster_centers):
    plt.scatter(data.iloc[:, 4], data.iloc[:, 10], c=data['cluster'], cmap='Spectral')
    plt.scatter(cluster_centers[:, 4], cluster_centers[:, 10], marker='o', s=50, color='red', label='Cluster Centers')
    plt.title('Mean Shift Clustering Results')
    plt.xlabel('List Price')
    plt.ylabel('Property Valuation')
    plt.show()


def subset_pairplot(data):
    subset_to_plot = ['list_price', 'property_valuation', 'past_3_years_bike_related_purchases', 'cluster']
    subset_df = data[subset_to_plot]
    sns.pairplot(subset_df, hue='cluster', palette='Spectral')
    plt.show()


def three_dim_scatter(data):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['list_price'], data['property_valuation'], data['past_3_years_bike_related_purchases'],
               c=data['cluster'], cmap='Spectral', s=50)
    # Set labels and title
    ax.set_xlabel('List Price')
    ax.set_ylabel('Property Valuation')
    ax.set_zlabel('Related Purchases in Past 3 Years')
    plt.show()


def unclustered_scatter(data):
    # Scatter Plot
    plt.scatter(data['list_price'], data['property_valuation'], alpha=0.7, edgecolors='w')
    mean_1 = data['property_valuation'].mean()
    mean_2 = data['list_price'].mean()
    # Plot the mean values
    plt.scatter(mean_1, mean_2, color='red', marker='o', s=50, label='Mean')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('', y=1.05)
    plt.show()

meanshift_clustering()
