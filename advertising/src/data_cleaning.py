import pandas as pd 
import numpy as np

def adjust_columns(data):
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.lower()

if __name__ == '__main__':
    data = pd.read_csv('../data/advertising.csv')
    adjust_columns(data)
    toggle = False
    if toggle:
        print(data.head())
        print(data.info())
        print(data.nunique())
    country_col = pd.get_dummies(data['country'], prefix='country')
    country_col = country_col.astype(int)
    data.drop(['city', 'ad_topic_line', 'timestamp', 'country'], axis=1, inplace=True)
    df = pd.concat([data, country_col], axis=1)
    df.to_csv('../data/advertising_clean.csv')

