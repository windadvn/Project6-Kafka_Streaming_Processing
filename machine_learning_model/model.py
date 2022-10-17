#!python3

import pickle
import pandas
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    
    df = pandas.read_csv('Fraud.csv')
    
    d_devices = pandas.get_dummies(df['device'])
    d_devices.insert(0,'device',df['device'])
    d_devices.drop_duplicates().to_json('dict_devices.json', orient='records')

    d_newbalanceDest = pandas.get_dummies(df['newbalanceDest'])
    d_newbalanceDest.insert(0,'newbalanceDest',df['newbalanceDest'])
    d_newbalanceDest.drop_duplicates().to_json('dict_newbalanceDest.json', orient='records')

    df = pandas.concat([df,pandas.get_dummies(df['device'])], axis=1).drop('device', axis=1)
    df = pandas.concat([df,pandas.get_dummies(df['newbalanceDest'])], axis=1).drop('newbalanceDest', axis=1)

    X = df.drop(['isFraud'], axis=1).values.tolist()
    y = df['isFraud'].values.tolist()
    
    model = LogisticRegression()
    model.fit(X,y)
    with open('modelFraud.pkl','wb') as file:
        pickle.dump(model, file)