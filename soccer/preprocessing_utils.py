import os
import warnings
import random 
import pandas as pd 
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

def get_outlier(df=None, column=None, weight=1.5):
  # target 값과 상관관계가 높은 열을 우선적으로 진행
  quantile_25 = np.percentile(df[column].values, 25)
  quantile_75 = np.percentile(df[column].values, 75)

  IQR = quantile_75 - quantile_25
  IQR_weight = IQR*weight
  
  lowest = quantile_25 - IQR_weight
  highest = quantile_75 + IQR_weight
  
  outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index
  return outlier_idx

def preprocessing(train, test, is_test=False):

    # Date col preprocessing
    train['year'] = train['date'].apply(lambda x : int(x[0:4]))
    train['month'] = train['date'].apply(lambda x : int(x[5:7]))
    train['day'] = train['date'].apply(lambda x : int(x[8:10]))
    test['year'] = test['date'].apply(lambda x : int(x[0:4]))
    test['month'] = test['date'].apply(lambda x : int(x[5:7]))
    test['day'] = test['date'].apply(lambda x : int(x[8:10]))
    train.drop(columns=['date'], inplace=True)

    #  match feature create 
    train['match'] = train['homeTeam'] + '-' + train['awayTeam']
    test['match'] = test['homeTeam'] + '-' + test['awayTeam']

    # hometeam / awayteam label encoding ( Test에서 성능 향상)
    train['home_win'] = train['result'].apply(lambda x: 1 if x=='H' else 0) 
    dic = {}
    for team in train['homeTeam'].unique():
        value = train[train['homeTeam'] == team]['home_win'].sum()
        dic[team] = value

    label_dic={}
    for idx, (team, _) in enumerate(sorted(dic.items(), key= lambda x: x[1])):
        label_dic[team] = idx

    train['homeTeam'] = train['homeTeam'].apply(lambda x: label_dic[x])
    train['awayTeam'] = train['awayTeam'].apply(lambda x: label_dic[x])
    test['homeTeam'] = test['homeTeam'].apply(lambda x: label_dic[x])
    test['awayTeam'] = test['awayTeam'].apply(lambda x: label_dic[x])

    # feature selection
    train = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)',  'home_win'])

    # 아래의 feature는 훈련에 사용하지 않는다.
    # matchID season date result goals(homeTeam) goals(awayTeam) homeTeam awayTeam	
    stats_columns = [
    'halfTimeGoals(homeTeam)',
    'halfTimeGoals(awayTeam)',
    'shots(homeTeam)',
    'shots(awayTeam)',
    'shotsOnTarget(homeTeam)',
    'shotsOnTarget(awayTeam)',
    'corners(homeTeam)',
    'corners(awayTeam)',
    'fouls(homeTeam)',
    'fouls(awayTeam)',
    'yellowCards(homeTeam)',
    'yellowCards(awayTeam)',
    'redCards(homeTeam)',
    'redCards(awayTeam)'
    ]

    # 모든 시즌에 대해서 진행해보았으나 test에서 202324 시즌 만 했을 때가 성능이 가장 좋았다. 
    # latest_two_season_df = train[train['season'] >= 202324]
    # pair_stats = latest_two_season_df.groupby('match')[stats_columns].mean().reset_index() 
    
    pair_stats = train.groupby('match')[stats_columns].mean().reset_index() 
    test_with_stats = test.merge(pair_stats, on='match', how='left')

    # 2부리그에서 1부리그로 처음 올라온 팀은 그전에 경기기록이 없다. 따라서, 평균으로 값 대체 
    test_with_stats.fillna(pair_stats[stats_columns].min(), inplace=True) # pair_stats mean
    if is_test == True:
        col_list = [col for col in train.columns if col != 'result']
        test = test_with_stats[col_list]
    else:
        test = test_with_stats[train.columns]

    # label encoding
    encoding_target = list(train.dtypes[train.dtypes == "object"].index)
    encoding_target.remove('result')
    for i in encoding_target:
        le = LabelEncoder()
        le.fit(train[i])
        train[i] = le.transform(train[i])
        
        # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
        for case in np.unique(test[i]):
            if case not in le.classes_: 
                le.classes_ = np.append(le.classes_, case)
        
        test[i] = le.transform(test[i])

    # outlier 제거 (성능향상) (ppt 정리)
    # oulier_idx_shotsHomeTeam = get_outlier(df=train, column='shots(homeTeam)', weight=1.5)
    # train.drop(oulier_idx_shotsHomeTeam, axis=0, inplace=True)
    # train.reset_index(drop=True, inplace=True)
    # oulier_idx_shotsAwayTeam = get_outlier(df=train, column='shots(awayTeam)', weight=1.5)
    # train.drop(oulier_idx_shotsAwayTeam, axis=0, inplace=True)
    # train.reset_index(drop=True, inplace=True)

    # Scaler (성능하락)
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)

    
    # split X and y (test and valid)

    if is_test:
        train_x = train.drop(columns=['result'])
        train_y = train['result']

        return train_x, train_y, test
    else:
        train_x = train.drop(columns=['result'])
        train_y = train['result']

        test_x = test.drop(columns=['result'])
        test_y = test['result']
        return train_x, train_y, test_x, test_y