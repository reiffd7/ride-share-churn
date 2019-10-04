import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer

def clean_it(data):
    '''
    data = path to data
    '''
    df = pd.read_csv(data)
    
    df = pd.get_dummies(df, columns=['city', 'phone', 'luxury_car_user'])
    df.drop(['luxury_car_user_False', 'city_Winterfell'], axis = 1, inplace=True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['month'] = df['last_trip_date'].map(lambda x: x.month)
    df['churn'] = np.where(df['month']>5, 0, 1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df['avg_rating_by_driver']= imp.fit_transform(df[['avg_rating_by_driver']]).ravel()
    df['avg_rating_of_driver']= imp.fit_transform(df[['avg_rating_of_driver']]).ravel()
    return df

def impute(df, columns):
    '''
    columns = list or array of columns in frame
    '''

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    for col in columns:
        df[col] = imp.fit_transform(df[[col]]).ravel()
    return df


def dummy_it(df, columns):
    '''
    columns = list or array of columns in dataframe
    '''
    df = pd.get_dummies(df, columns = columns)
    return df

def drop_col(df, columns):
    df.drop(columns, axis=1, inplace=True)
    return df

def datetime_it(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df


# def drop_outlier(df, columns, condition):

if __name__ == "__main__":


    # df = pd.read_csv('../data/churn.csv')

    df = pd.read_csv('../data/churn.csv')

    # impute cols
    impute_cols = ['avg_rating_of_driver', 'avg_rating_by_driver']
    df = impute(df, impute_cols)

    # datetime cols
    date_cols = ['signup_date', 'last_trip_date']
    df = datetime_it(df, date_cols)

    # create month and days of january columns
    df['month'] = df['last_trip_date'].map(lambda x: x.month)
    df['total_days_january'] = df['signup_date'].map(lambda x: x.day)

    # create target column using new month column
    df['churn'] = np.where(df['month']>5, 0, 1)

    # dummy columns
    dummy_cols = ['city', 'phone', 'luxury_car_user']
    df = dummy_it(df, dummy_cols)

    # drop columns
    drop_cols = ['luxury_car_user_False', 'city_Winterfell', 'signup_date', 'last_trip_date', 'month']
    df = drop_col(df, drop_cols)

    df.to_csv('cleaned_data.py')