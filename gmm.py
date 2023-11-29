import pandas as pd

import import_data

data = import_data.get_dataframe('segmentation_data.csv')

# training gaussian mixture model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)

#predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)

# Output data to CSV
data['Cluster'] = labels
data.to_csv('outputs/gmm_output.csv', sep=',', encoding='utf-8')


# https://gist.github.com/aishwarya-singh25
# https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/#?&utm_source=coding-window-blog&source=coding-window-blog
# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html
