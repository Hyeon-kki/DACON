import os
import warnings
import random 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder


def preprocessing(train, test):

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

    train['match'] = train['homeTeam'] + '-' + train['awayTeam']
    pair_stats = train.groupby('match')[stats_columns].mean().reset_index() # match mean

    # test_with_stats
    test['match'] = test['homeTeam'] + '-' + test['awayTeam']
    test_with_stats = test.merge(pair_stats, on='match', how='left')

    # 2부리그에서 1부리그로 처음 올라온 팀은 그전에 경기기록이 없다. 따라서, 평균으로 값 대체 
    test_with_stats.fillna(pair_stats[stats_columns].mean(), inplace=True) # pair_stats mean
    test = test_with_stats[train.columns]

    # label encoding
    encoding_target = list(train.dtypes[train.dtypes == "object"].index)
    for i in encoding_target:
        le = LabelEncoder()
        le.fit(train[i])
        train[i] = le.transform(train[i])
        
        # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
        for case in np.unique(test[i]):
            if case not in le.classes_: 
                le.classes_ = np.append(le.classes_, case)
        
        test[i] = le.transform(test[i])
    return train, test