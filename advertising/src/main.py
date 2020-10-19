import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == '__main__':
    df = pd.read_csv('../data/advertising_clean.csv')
    X = df.drop(['clicked_on_ad', 'Unnamed: 0'], axis=1)
    y = df['clicked_on_ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)
    print('Model Results: \n', classification_report(y_test, y_pred))

    '''
    Model Results: 
                   precision    recall  f1-score   support

            0          0.85      0.96      0.90       146
            1          0.96      0.84      0.89       154

    accuracy                               0.90       300
    macro avg          0.90      0.90      0.90       300
    weighted avg       0.90      0.90      0.90       300
    '''
