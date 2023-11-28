"""
    Author: Matthew Vallance 001225832
    Purpose: Writing my own K-Means algorithm from tutorial: https://www.youtube.com/watch?v=iNlZ3IU5Ffw
    Date: 10/11/23
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('data/force2020_data_unsupervised_learning.csv', index_col = 'DEPTH_MD')

# Display the data
print('dataframe after load:')
print(df)

# Apply dropna to dataframe (df) - This removes all the NaN values as algorithm cant handle these values
df.dropna(inplace=True)

# Display the data
print('After dropna applied:')
print(df)

# df.describe - This shows
print(df.describe())