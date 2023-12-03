import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def remove_encode():
    # Reading CSV file into dataframe
    data = pd.read_csv('99Bikers_REMOVED_ENCODED.csv')

    # Removing unnecessary columns -'transaction_id', 'product_id', 'customer_id', 'product_first_sold_date',
    # 'first_name', 'last_name', 'job_title', 'tenure'
    #columns_to_remove = ['transaction_date', 'order_status', 'standard_cost', 'customer_id', 'job_industry_category',
    #                     'deceased_indicator']
    #data = data.drop(columns=columns_to_remove, axis=1)

    # Changing binary features to 1 or 0
    binary_features = ['online_order', 'owns_car']
    for feature in binary_features:
        data[feature] = data[feature].replace({True: 1, False: 0, '': 0, 'Yes': 1, 'No': 0})

    # Identifying categorical features and reassigning values
    categorical_features = ['brand', 'product_line', 'product_size', 'gender', 'wealth_segment']
    feature_dicts = [{' ': 0, 'Solex': 1, 'Trek Bicycles': 2, 'OHM Cycles': 3, 'Norco Bicycles': 4, 'Giant Bicycles': 5,
                      'WeareA2B': 6},
                     {' ': 0, 'Standard': 1, 'Road': 2, 'Mountain': 3, 'Touring': 4},
                     {' ': 0, 'small': 1, 'medium': 1, 'large': 2},
                     {'F': 0, 'Femal': 0, 'Female': 0, 'M': 1, 'Male': 1, 'U': 1, ' ': 1},
                     {'Mass Customer': 0, 'Affluent Customer': 1, 'High Net Worth': 2}]

    # Replacing categorical features with values assigned above
    for feature, feature_dict in zip(categorical_features, feature_dicts):
        data[feature] = data[feature].replace(feature_dict)

    data = data.dropna()    # Remove all rows with NaN vals

    data.to_csv('99Bikers_REMOVED_ENCODED.csv', index=False)


def get_unique():
    data = pd.read_csv('99Bikers_REMOVED.csv')
    unique_values = data[('wealth_segment')].unique()
    print(unique_values)


def scaler():
    # Loading dataset
    df = pd.read_csv('99Bikers_REMOVED_ENCODED.csv', header=0)

    columns_to_scale = ['list_price', 'age', 'past_3_years_bike_related_purchases']
    scaler = MinMaxScaler(feature_range=(0, 10))
    scaled_columns = scaler.fit_transform(df[columns_to_scale])
    df_scaled = pd.DataFrame(scaled_columns, columns=columns_to_scale)
    df[columns_to_scale] = df_scaled
    df.to_csv('99Bikers_REMOVED_ENCODED_SCALED.csv', index=False)


scaler()
