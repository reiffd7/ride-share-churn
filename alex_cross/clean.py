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




if __name__ == "__main__":


    # df = pd.read_csv('../data/churn.csv')

    df = clean_it('../data/churn.csv')