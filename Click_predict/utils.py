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
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce

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
    submission = pd.read_csv("/home/workspace/DACON/Click_predict/data/sample_submission.csv")

    return train, test, submission
    
''' Optuna '''
# def objective(trial, train):

#   param = {"n_estimators": trial.suggest_int("n_estimators:", 100, 1000),
#            "max_depth": trial.suggest_int("max_depth", 6, 30),
#            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
#            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3),
#            'lambda': trial.suggest_float('lambda', 1e-3, 0.1),
#            'alpha': trial.suggest_float('alpha', 1e-3, 1.0),
#            'min_child_weight': trial.suggest_int('min_child_weight', 2, 50)}
  
#   auc_score = []
#   X_train = train.drop(columns='Click')
#   y_train = train['Click']
#   X_train.reset_index(drop = True, inplace=True)
#   y_train.reset_index(drop = True, inplace=True)

#   model = XGBClassifier(random_state=42, tree_method= 'gpu_hist', **param)
#   S_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#   for iter, (train_index, test_index) in enumerate(S_kfold.split(X_train, y_train)):
#       x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
#       y_train, y_test = y_train.iloc[train_index], y_train.iloc[test_index]
#       model.fit(x_train, y_train, eval_metric='auc')
#       valid_pred = model.predict_proba(x_test)
#       score = roc_auc_score(y_test, valid_pred[:, 1])
#       auc_score.append(score)
      
#   avg_score = sum(auc_score)/3
#   return avg_score

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
    soft_voting_value = soft_voting_value / k

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

def groupby_mean(train_df, valid_df, test_df, group_col, numeric_cols):
    # Dictionary to store the mean values for each numeric column
    mean_dict = {}
    
    for numeric_col in tqdm(numeric_cols):
        # Calculate mean for train data
        agg = train_df.groupby(group_col)[numeric_col].mean()
        mean_dict[numeric_col] = agg
        
        # Map the calculated means to train, valid, and test data
        train_df[f'{group_col}_{numeric_col}_mean'] = train_df[group_col].map(agg)
        valid_df[f'{group_col}_{numeric_col}_mean'] = valid_df[group_col].map(agg)
        test_df[f'{group_col}_{numeric_col}_mean'] = test_df[group_col].map(agg)
    
    return train_df, valid_df, test_df

# Optuna를 사용하여 하이퍼파라미터 튜닝
# def objective(trial):

#     # LGBM 하이퍼파라미터
#     lgbm_params = {
#         "n_estimators": trial.suggest_int("n_estimators:", 1000, 3000),
#         "max_depth": trial.suggest_int("max_depth", 6, 30),
#         #"subsample": trial.suggest_float("subsample", 0.3, 1.0),
#         "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3)}
    

#     # XGBoost 하이퍼파라미터
#     xgb_params = {
#         "n_estimators": trial.suggest_int("n_estimators:", 1000, 3000),
#         "max_depth": trial.suggest_int("max_depth", 6, 30),
#         #"subsample": trial.suggest_float("subsample", 0.3, 1.0),
#         "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3)}

#     # k-NN 하이퍼파라미터
#     knn_params = {
#         'n_neighbors': trial.suggest_int('knn_n_neighbors', 100, 1000)
#     }

#     # 로지스틱 회귀 하이퍼파라미터
#     lr_params = {
#         'C': trial.suggest_float('lr_C', 0.0001, 100, log=True)
#     }

#     # 개별 모델 정의
#     lgbm = LGBMClassifier(device='gpu',**lgbm_params)
#     xgb = XGBClassifier(random_state=42, tree_method= 'gpu_hist', **xgb_params)
#     knn = KNeighborsClassifier(**knn_params)
#     lr = LogisticRegression(**lr_params)

#     # 스태킹 분류기 정의
#     stacking_clf = StackingClassifier(
#         estimators=[
#             ('lgbm', lgbm),
#             ('xgb', xgb),
#             ('knn', knn),
#             ('lr', lr)
#         ],
#         final_estimator=LogisticRegression()
#     )

#     # 모델 학습
#     stacking_clf.fit(X_train, y_train)

#     # 검증 데이터에 대한 예측
#     y_valid_pred = stacking_clf.predict_proba(X_valid)[:, 1]
#     return roc_auc_score(y_valid, y_valid_pred)

# # Optuna 스터디 정의 및 최적화
# sampler = TPESampler(seed=42)
# study = optuna.create_study(direction='maximize', sampler=sampler)
# study.optimize(objective, n_trials=10)

# # 최적 하이퍼파라미터 출력
# print("Best trial:")
# trial = study.best_trial
# print(f"  AUC: {trial.value:.4f}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

def preprocessing(X_train, X_valid, y_train, y_valid, test, is_test = False):

    ''' Feature Selection '''
    print('Feature Selection')
    X_train.drop(columns = ['ID'], inplace = True)
    X_valid.drop(columns = ['ID'], inplace = True)
    numeric_col = X_train.select_dtypes(include=['Float64', 'Float32', 'int64', 'int32', 'int16', 'int8']).columns
    if not is_test:
        numeric_col = numeric_col.drop('Click')

    # target_col = ["F03", "F08", "F15", "F16", "F17", "F28", "F31"]
    # encoder_1 =  ce.TargetEncoder()
    # X_train[target_col] = encoder_1.fit_transform(X_train[target_col], y_train)
    # X_valid[target_col] = encoder_1.transform(X_valid[target_col], y_valid)
    # test[target_col] = encoder_1.transform(test[target_col])

    object_col = X_train.select_dtypes(include=['object']).columns

    ''' 빈도 인코딩 '''
    print('Start Frequency')
    # 각 범주형 변수의 빈도수 계산 및 인코딩
    encoder_2 =  ce.CountEncoder()
    X_train[object_col] = encoder_2.fit_transform(X_train[object_col], y_train)
    X_valid[object_col] = encoder_2.transform(X_valid[object_col], y_valid)
    test[object_col] = encoder_2.transform(test[object_col])
    
    ''' F17 '''
    X_train = groupby_mean(X_train, 'F17', X_train.columns)
    X_valid = groupby_mean(X_valid, 'F17', X_train.columns)
    test = groupby_mean(test, 'F17', X_train.columns)

    ''' 결측치 처리 ''' 
    print('Missing Value')
    print("---------------- Start MissingValue ----------------")
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)
    test = test.fillna(0)
    
    X_train = reduce_mem_usage(X_train)
    X_valid = reduce_mem_usage(X_valid)
    test = reduce_mem_usage(test)

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
    # print("---------------- Change Dtype ----------------")
    # df = reduce_mem_usage(df)
    # object_col = df.select_dtypes(include=['object']).columns
    # df[object_col] = df[object_col].astype('category')

    return X_train, X_valid, y_train, y_valid, test


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