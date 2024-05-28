import pandas as pd
import numpy as np
import random
import os
import duckdb
import polars as pl
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import MinMaxScaler

def feature_summary(df):
    temp = 'Dataframe Size'
    print(f'----------------------------{temp:^16}----------------------------')
    print(f'num: {len(df)}')

    temp = 'Dtype Size'
    type_idx = df.dtypes.value_counts().index
    type_num = df.dtypes.value_counts().values
    print(f'----------------------------{temp:^16}----------------------------')
    for type, num in (zip(type_idx, type_num)):
        print(f'Type {type} : {num}')
    feature_summary = pd.DataFrame(df.dtypes, columns=["Data Type"]) # dtype
    feature_summary.reset_index(inplace=True)
    feature_summary.rename(columns={'index': 'Feature Name'}, inplace=True) 
    feature_summary['Nunique'] = df.nunique().values 
    feature_summary['NullValue'] = df.isnull().sum().values
    feature_summary['NullValue Ratio'] = (df.isnull().sum().values / len(df))*100
    feature_summary['value_1'] = df.loc[0].values
    feature_summary['value_2'] = df.loc[1].values
    feature_summary['value_3'] = df.loc[2].values
    return feature_summary

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_data_polars():
    train = pd.read_csv('/home/workspace/DACON/Click_predict/data/train.csv') 
    test = pd.read_csv('/home/workspace/DACON/Click_predict/data/test.csv')
    return train, test

def load_data():
    train = pd.read_csv('/home/workspace/DACON/Click_predict/data/train.csv') 
    test = pd.read_csv('/home/workspace/DACON/Click_predict/data/test.csv')
    return train, test

def to_pandas(train, test): #판다스로 바꾸기. 문자형 변수들 카테고리로 변환
    train = train.to_pandas()
    test = test.to_pandas()

    return train, test

def Kfold(model, k, X_train, Y_train, test, is_test = False):

    valid_arrays = []
    test_arrays = []
    soft_voting_value = np.zeros((len(test)))

    S_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X_train.reset_index(drop = True, inplace=True)
    Y_train.reset_index(drop = True, inplace=True)

    for iter, (train_index, test_index) in enumerate(S_kfold.split(X_train, Y_train)):
        x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
        model.fit(x_train, y_train, eval_metric='AUC')
        valid_pred = model.predict_proba(x_test)
        valid_arrays.append(valid_pred[:, 1])
        score = roc_auc_score(y_test, valid_pred[:, 1])
        print(f'---------------- {iter+1} fold의 Acc: {score} ----------------')

        if is_test:
            ''' soft voting '''
            test_pred = model.predict_proba(test)
            test_arrays.append(test_pred[:, 1])
            for value in test_arrays:
                print(soft_voting_value)
                soft_voting_value += value
                print(soft_voting_value)
    soft_voting_value /= k

    return valid_arrays, test_arrays, soft_voting_value

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

def filter_cols(df): #결측치가 35%이상 되는 피처들 제외
    for col in df.columns:
        if col not in ["ID"]:
            missing_percentage = df[col].null_count() / len(df) * 100
            if missing_percentage > 35:
                print(col)
                df = df.drop(col)
    print("____________________________________________")
    for col in df.columns: #클래스 수가 1 혹은 200이 넘는 카테고리 변수들 제거
        if (col not in ["ID"]) & (df[col].dtype == pl.String):
            freq = df[col].n_unique()
            if (freq == 1) | (freq > 4538541):
                print(col)
                df = df.drop(col)        
    return df

def reduce_mem_usage(df): #판다스에 적용할 것
    
    # 메모리 최적화
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def fill_missing_values(df):
    """
    폴라스 데이터프레임에서 열을 순회하며,
    문자형 열의 결측치를 "nan"으로 채우고,
    수치형 열의 결측치를 0으로 채움
    """
    for column in df.columns:
        if df[column].dtype == 'String':  # 문자형 열인 경우
            df = df.with_columns(pl.col(column).fill_null(np.nan))
        else:  # 수치형 열인 경우
            df = df.with_columns(pl.col(column).fill_null(0))
    return df

def feature_category(df, feature_list, treshold):
    for col in tqdm(feature_list):
        Ndic = df[col].value_counts()
        df[f'{col}_new'] = df[col].apply(lambda x: "etc" if Ndic[x] < treshold else x)
        df[f'{col}_new'] = df[f'{col}_new'].astype('category')
    return df

def groupby_mean(df, group_col, numeric_cols):
    for numeric_col in numeric_cols:
        agg = df.groupby(group_col)[numeric_col].mean()
        df[f'{group_col}_{numeric_col}_mean'] = df[group_col].map(agg)
    return df

def preprocessing(df, is_test = False):

    ''' Feature Selection '''
    df.drop(columns = ['ID'], inplace = True)
    numeric_col = df.select_dtypes(include=['Float64', 'Float32', 'int64', 'int32', 'int16', 'int8']).columns
    if not is_test:
        numeric_col = numeric_col.drop('Click')
    object_col = df.select_dtypes(include=['object']).columns

    ''' 결측치 처리 ''' 
    print("---------------- Start MissingValue ----------------")
    for col in tqdm(numeric_col):
        df[col] = df[col].fillna(0)
    for col in tqdm(object_col):
        df[col] = df[col].fillna('NaN')

    ''' groupby 실험 '''
    # df = groupby_mean(df, "F15",  numeric_col)

    ''' log 변환 (실험1)''' # 성능에 변화없음 
    # df['F29_log'] = np.log(df['F29'] + 1) 

    ''' 추가 '''
    # df['F09_F07'] = df['F09']+df['F07'] # 0.778695906242
    # df['F09_F25'] = df['F09']+df['F25'] # 0.7787953
    # df['F2729_add'] = df['F27']+df['F29'] # 성능 향상됨 (사용하기)
    # df['F2729_mul'] = df['F27']*df['F29'] # 


    ''' 중요 Feature 카테고리 '''
    # print("---------------- Start Category ----------------")

    # df = feature_category(df, obj_col, 5)


    ''' 메모리 사용량 줄임 '''
    print("---------------- Change Dtype ----------------")
    df = reduce_mem_usage(df)
    object_col = df.select_dtypes(include=['object']).columns
    df[object_col] = df[object_col].astype('category')

    return df 


    # 실험1 (결측값 수가 같은 것끼리 동일한 결측값 채우기) (성능 하락)
    # missing_same_list=[]
    # missing_same_list.append(["F01", "F02", "F05", "F10", "F12", "F34"]) # Object
    # missing_same_list.append(["F03", "F15", "F20", "F26" ]) # Object
    # missing_same_list.append(["F19", "F33"]) # Float
    # missing_same_list.append(["F27", "F29"]) # Float
    # O_iter,N_iter = 0, 0 # 둘 다 2까지 갈 듯
    # for list in missing_same_list:
    #     feature_type = df[list[0]].dtype
    #     for feature in list:
    #         if feature_type == 'object':
    #             df[feature].fillna('NaN'+str(O_iter))
    #             O_iter += 1
    #         else:
    #             df[feature].fillna(N_iter)
    #             N_iter += 1
    # for col in numeric_col:
    #     df[col] = df[col].fillna(-1)
    # for col in object_col:
    #     df[col] = df[col].fillna('NaNK')