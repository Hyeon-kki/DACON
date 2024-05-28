import pandas as pd
import numpy as np
import random
import os
import duckdb
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import matplotlib.dates as mdates

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

def datetime_process(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name])
    df['Y'] = df[col_name].dt.year
    df['M'] = df[col_name].dt.month
    df['D'] = df[col_name].dt.day
    df['Day_name'] = df[col_name].dt.day_name()
    df['hour'] = df[col_name].dt.hour
    # df.drop(columns=[col_name], inplace = True)
    return df

def ma_ema_feature(df, col_name, window_size):
    df[f'{col_name}_ma_{window_size}'] = df[col_name].rolling(window = window_size).mean()
    df[f'{col_name}_ema_{window_size}'] = df[col_name].ewm(span = window_size).mean()
    
    # 시각화
    # y축 범위를 각 그래프의 최대, 최소값으로 설정
    y_min = min(df[f'{col_name}'].min(), df[f'{col_name}_ma_{window_size}'].min(), df[f'{col_name}_ema_{window_size}'].min())
    y_max = max(df[f'{col_name}'].max(), df[f'{col_name}_ma_{window_size}'].max(), df[f'{col_name}_ema_{window_size}'].max())

    plt.figure(figsize=(12, 4))

    # 가격, MA, EMA 그래프 그리기
    plt.plot(df['Time'], df[f'{col_name}'], label='Original', alpha=0.5)
    plt.plot(df['Time'],  df[f'{col_name}_ma_{window_size}'], label=f'MA-{window_size}')
    plt.plot(df['Time'],  df[f'{col_name}_ema_{window_size}'], label=f'EMA-{window_size}')

    # x축과 y축의 범위를 설정
    plt.xlim([df['Time'].min(), df['Time'].max()])
    plt.ylim([y_min, y_max])

    # x축을 날짜 형식으로 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()

    # 그래프의 제목과 레이블을 설정
    plt.title(f'Moving Averages and Exponential Moving Averages of {col_name}')
    plt.xlabel('Date')
    plt.ylabel(f'{col_name}')

    # 범례를 표시
    plt.legend()

    # 그래프를 출력
    plt.show()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

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

def Check_data(train, test, col_name: str) -> None:
    ''' 기간 확인하는 함수 '''
    train_min_date = train[col_name].min()
    train_max_date = train[col_name].max()

    test_min_date = test[col_name].min()
    test_max_date = test[col_name].max()

    print(f"train 데이터 기간 : {train_min_date} ~ {train_max_date}")
    print(f"test 데이터 기간 : {test_min_date} ~ {test_max_date}")

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

    
    ''' 중요 Feature 카테고리 '''
    print("---------------- Start Category ----------------")
    important_features = ["F09", "F39", "F13", "F20", "F37"]
    df = feature_category(df, important_features, 5)


    # F01_Ndic = df['F01'].value_counts()
    # df['F01'] = df['F01'].apply(lambda x: "etc" if F01_Ndic[x] < 100 else x)

    ''' 메모리 사용량 줄임 '''
    print("---------------- Change Dtype ----------------")
    df = reduce_mem_usage(df)
    df[object_col] = df[object_col].astype('category')

    return df 

