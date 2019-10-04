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
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['month'] = df['last_trip_date'].map(lambda x: x.month)
    df['churn'] = np.where(df['month']>5, 0, 1)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df['avg_rating_by_driver']= imp.fit_transform(df[['avg_rating_by_driver']]).ravel()
    df['avg_rating_of_driver']= imp.fit_transform(df[['avg_rating_of_driver']]).ravel()
    return df

def drop_cols(df, cols_to_drop):
    '''
    inputs: data frame
            list of columns to drop

    returns:
            dataframe with dropped columns
    '''
    for col in cols_to_drop:
        df.drop([col], axis=1, inplace=True)
    return df


def col_to_date(df, cols_to_convert):
    '''
    inputs: pandas data frame
            list of columns to convert to datetime

    returns:
            dataframe with converted columns
    '''
    for col in cols_to_drop:
        df.drop([col], axis=1, inplace=True)
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
    df = pd.get_dummies(df, columns)

    return df

# def drop_outlier(df, columns, condition):

if __name__ == "__main__":


    # df = pd.read_csv('../data/churn.csv')

    df = clean_it('../data/churn.csv')