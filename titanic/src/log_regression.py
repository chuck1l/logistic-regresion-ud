import pandas as pd  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # First model without pclass as dummies
    train_data = pd.read_csv('../data/clear_titanic_train.csv')
    test_data = pd.read_csv('../data/clear_titanic_test.csv')
    # Second model with pclass as dummies
    train_data_pclass = pd.read_csv('../data/clear_titanic_train_plcass_dum.csv')
    test_data_pclass = pd.read_csv('../data/clear_titanic_test_pclass_dum.csv')

    X1_train, X1_val, y1_train, y1_val = train_test_split(train_data.drop('Survived', axis=1), 
                                                        train_data['Survived'],
                                                        test_size=0.30,
                                                        random_state=101)
    X1_test = test_data.copy()
    
    X2_train, X2_val, y2_train, y2_val = train_test_split(train_data_pclass.drop('Survived', axis=1),
                                                        train_data_pclass['Survived'],
                                                        test_size=0.30,
                                                        random_state=101)
    X2_test = test_data_pclass.copy()

    logmodel1 = LogisticRegression(max_iter=2000)
    logmodel1.fit(X1_train, y1_train)
    logmodel2 = LogisticRegression(max_iter=2000)
    logmodel2.fit(X2_train, y2_train)
    y_pred1 = logmodel1.predict(X1_val)
    y_pred2 = logmodel2.predict(X2_val)

    print('The Resulst From Model 1: \n', classification_report(y1_val, y_pred1))
    print('The Resulst From Model 2: \n', classification_report(y2_val, y_pred2))

    