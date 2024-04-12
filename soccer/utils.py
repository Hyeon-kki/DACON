import pandas as pd

def load_train_valid():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    # sm의 Logit 모듈 쓸려고 전처리 한것
    # train = train[train['result'] != 'D']
    # train.reset_index(inplace=True, drop= True)
    test = train[-88:][['season', 'homeTeam', "awayTeam", 'date']]
    train = train[:-88]
    valid_y = train[-88:]['result']

    # train date 처리
    train['year'] = train['date'].apply(lambda x : int(x[0:4]))
    train['month'] = train['date'].apply(lambda x : int(x[5:7]))
    train['day'] = train['date'].apply(lambda x : int(x[8:10]))
    train.drop(columns=['date'], inplace=True)

    # valid date 처리
    test['year'] = test['date'].apply(lambda x : int(x[0:4]))
    test['month'] = test['date'].apply(lambda x : int(x[5:7]))
    test['day'] = test['date'].apply(lambda x : int(x[8:10]))
    test.drop(columns=['date'], inplace=True)

    #  match feature create 
    train['match'] = train['homeTeam'] + '-' + train['awayTeam']
    test['match'] = test['homeTeam'] + '-' + test['awayTeam']

    # hometeam / awayteam label encoding
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
    test = test.drop(columns=['home_win'])

    train_x = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)', 'result', 'home_win'])
    train_y = train['result']

    return train_x, train_y, test, valid_y 

def load_train_test():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    test = pd.read_csv('/home/workspace/DACON/soccer/Data/test.csv')

    # train date 처리
    train['year'] = train['date'].apply(lambda x : int(x[0:4]))
    train['month'] = train['date'].apply(lambda x : int(x[5:7]))
    train['day'] = train['date'].apply(lambda x : int(x[8:10]))
    train.drop(columns=['date'], inplace=True)

    # valid date 처리
    test['year'] = test['date'].apply(lambda x : int(x[0:4]))
    test['month'] = test['date'].apply(lambda x : int(x[5:7]))
    test['day'] = test['date'].apply(lambda x : int(x[8:10]))
    test.drop(columns=['date'], inplace=True)
    
    #  match feature create 
    train['match'] = train['homeTeam'] + '-' + train['awayTeam']
    test['match'] = test['homeTeam'] + '-' + test['awayTeam']

    # hometeam / awayteam label encoding
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
    test = test.drop(columns=['home_win'])

    train_x = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)', 'result', 'home_win'])
    train_y = train['result']

    return train_x, train_y, test

