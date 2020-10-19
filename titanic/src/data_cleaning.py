import pandas as pd 
import numpy as np

def input_null_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age  

if __name__ == '__main__':
    train = pd.read_csv('../data/titanic_train.csv')
    test = pd.read_csv('../data/titanic_test.csv')
    # Create average age for each class to fill na's
    grouped_age = train.groupby('Pclass')['Age'].mean()
    age1 = int(grouped_age.iloc[0])
    age2 = int(grouped_age.iloc[1])
    age3 = int(grouped_age.iloc[2])
    print(age1, age2, age3)
    train['Age'] = train[['Age', 'Pclass']].apply(input_null_age, axis=1)
    test['Age'] = test[['Age', 'Pclass']].apply(input_null_age, axis=1)
    # Drop the cabin column, too many null values for signal
    train.drop('Cabin', axis=1, inplace=True)
    test.drop('Cabin', axis=1, inplace=True)
    # Drop remaining NaN from each train and test
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    # Get dummies for categorical features
    sex_train = pd.get_dummies(train['Sex'], prefix='sex', drop_first=True)
    sex_test = pd.get_dummies(test['Sex'], prefix='sex', drop_first=True)
    embark_train = pd.get_dummies(train['Embarked'], prefix='embark', drop_first=True)
    embark_test = pd.get_dummies(test['Embarked'], prefix='embark', drop_first=True)
    train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    train = pd.concat([train,sex_train,embark_train],axis=1)
    test = pd.concat([test,sex_test,embark_test],axis=1)
    # Save the new data frames
    train.to_csv('../data/clear_titanic_train.csv')
    test.to_csv('../data/clear_titanic_test.csv')
    # dummify the Pclass to see a difference in model error
    pclass_train = pd.get_dummies(train['Pclass'],prefix='class', drop_first=True)
    pclass_test = pd.get_dummies(test['Pclass'],prefix='class', drop_first=True)
    train.drop(['Pclass'],axis=1,inplace=True)
    test.drop(['Pclass'],axis=1,inplace=True)
    train = pd.concat([train,pclass_train],axis=1)
    test = pd.concat([test,pclass_test],axis=1)
    # Save the new data frames
    train.to_csv('../data/clear_titanic_train_plcass_dum.csv')
    test.to_csv('../data/clear_titanic_test_pclass_dum.csv')
    print(train.head())
