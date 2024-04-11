import pandas as pd

def load_train_valid():
    train = pd.read_csv('/home/workspace/DACON/soccer/Data/train.csv')
    valid_x = train[-88:][['season', 'homeTeam', "awayTeam", 'date']]
    train = train[:-88]
    valid_y = train[-88:]['result']

    # train date 처리
    train['year'] = train['date'].apply(lambda x : int(x[0:4]))
    train['month'] = train['date'].apply(lambda x : int(x[5:7]))
    train['day'] = train['date'].apply(lambda x : int(x[8:10]))
    train.drop(columns=['date'], inplace=True)

    # valid date 처리
    valid_x['year'] = valid_x['date'].apply(lambda x : int(x[0:4]))
    valid_x['month'] = valid_x['date'].apply(lambda x : int(x[5:7]))
    valid_x['day'] = valid_x['date'].apply(lambda x : int(x[8:10]))
    valid_x.drop(columns=['date'], inplace=True)

    train_x = train.drop(columns=['matchID', 'goals(homeTeam)', 'goals(awayTeam)', 'result'])
    train_y = train['result']
    
    return train_x, train_y, valid_x, valid_y 

