import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def load_data(df):
    train = pd.read_csv('/home/workspace/DACON/Click_predict/data/train.csv') 
    test = pd.read_csv('/home/workspace/DACON/Click_predict/data/test.csv')
    return train, test

def load_data(df):
    train = pd.read_csv('/home/workspace/DACON/Click_predict/data/train.csv') 
    test = pd.read_csv('/home/workspace/DACON/Click_predict/data/test.csv')
    return train, test

def Kfold(model, X_train, y_train, k):
    Acc_list = []
    S_kfold = StratifiedKFold(n_splits=k)

    for iter, (train_index, test_index) in enumerate(S_kfold.split(X_train, y_train)):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = np.round(accuracy_score(y_test, pred), 4) 
        Acc_list.append(accuracy)
        print(f'{iter} fold의 Acc: {accuracy}')

    print(f'Aver Acc: {sum(Acc_list)/k}')
    return Acc_list

def sampling(path):
    con = duckdb.connect()

    # 전체 데이터에서 0 -> 70% 1 -> 30% 
    df = con.query(f"""(SELECT *
                            FROM read_csv_auto('{path}')
                            WHERE Click = 0
                            ORDER BY random()
                            LIMIT 10000000)
                            UNION ALL
                            (SELECT *
                            FROM read_csv_auto('{path}')
                            WHERE Click = 1
                            ORDER BY random()
                            LIMIT 4000000)""").df()
    
    con.close()
    return df

def preprocessing(df):
    
    return df 