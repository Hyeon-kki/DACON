import pandas as pd
import numpy as np
import random
import os
import duckdb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_data():
    train = pd.read_csv('/home/workspace/DACON/Click_predict/data/train.csv') 
    test = pd.read_csv('/home/workspace/DACON/Click_predict/data/test.csv')
    return train, test

def Kfold(model, k, X_train, Y_train):
    roc_auc_score_list = []
    pred_list = []
    S_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X_train.reset_index(drop = True, inplace=True)
    Y_train.reset_index(drop = True, inplace=True)
    for iter, (train_index, test_index) in enumerate(S_kfold.split(X_train, Y_train)):
        x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
        model.fit(x_train, y_train, eval_metric='AUC')
        pred = model.predict_proba(x_test)
        pred_list.append(pred)
        score = roc_auc_score(y_test, pred[:, 1])
        roc_auc_score_list.append(score)
        print(f'---------------- {iter+1} fold의 Acc: {score} ----------------')

    print(f'---------------- Aver Acc: {sum(roc_auc_score_list)/k} ----------------')
    return pred_list, roc_auc_score_list, k

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

def preprocessing(df, is_test = False):

    ''' Feature Selection '''
    df.drop(columns = ['ID'], inplace = True)
    numeric_col = df.select_dtypes(include=['Float64', 'int64']).columns
    if not is_test:
        numeric_col = numeric_col.drop('Click')
    object_col = df.select_dtypes(include=['object']).columns

    ''' 결측치 처리 ''' 
    for col in numeric_col:
        df[col] = df[col].fillna(0)
    for col in object_col:
        df[col] = df[col].fillna('NaN')

    ''' 메모리 사용량 줄임 '''
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype('int64')
    df[object_col] = df[object_col].astype('category')
    
    return df 

