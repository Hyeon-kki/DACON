import os
import warnings
import random 
import pandas as pd 
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

def preprocessing(train_x, test_x):

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

    ''' test feature interpolation '''
    latest_two_season_df = train_x[train_x['season'] >= 202324]
    pair_stats = latest_two_season_df.groupby('match')[stats_columns].mean().reset_index() 
    test_x_with_stats = test_x.merge(pair_stats, on='match', how='left')

    # 2부리그에서 1부리그로 처음 올라온 팀은 그전에 경기기록이 없다. 따라서, 평균으로 값 대체 
    test_x_with_stats.fillna(pair_stats[stats_columns].mean(), inplace=True) # pair_stats mean
    test_x = test_x_with_stats[train_x.columns]

    # label encoding
    encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
    print(encoding_target)
    for i in encoding_target:
        le = LabelEncoder()
        le.fit(train_x[i])
        train_x[i] = le.transform(train_x[i])
        
        # test_x 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
        for case in np.unique(test_x[i]):
            if case not in le.classes_: 
                le.classes_ = np.append(le.classes_, case)
        
        test_x[i] = le.transform(test_x[i])

    # Scaler 
    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.transform(test_x)

    return train_x, test_x