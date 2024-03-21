from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm


class BaselineModel():
    def __init__(self, threshold = 0.5) -> None:
        super(BaselineModel, self).__init__() # 이름 정의하는 것이다. 
        self.threshold = threshold
        self.vocabulary = set()
        self.vectorizer = None

    def get_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(vocabulary=list(self.vocabulary))

    def fit(self, code):
        temp_vectorizer = CountVectorizer()
        temp_vectorizer.fit(code)
        self.vocabulary.update(temp_vectorizer.get_feature_names_out())
        self.vectorizer = self.get_vectorizer()
    
    def predict_prob(self, code1, code2) -> np.array:
        code1_vectors = self.vectorizer.transform(code1)
        code2_vectors = self.vectorizer.transform(code2)

        preds = []

        for code1_vec, code2_vec in tqdm(zip(code1_vectors ,code2_vectors)):
            preds.append(cosine_similarity(code1_vec, code2_vec))
        
        preds = np.reshape(preds, len(preds))

        return preds
    
    def predict(self, code1, code2) -> np.array:
        preds = self.predict_proba(code1, code2) # 유사도 계산합니다.
        preds = np.where(preds>self.threshold, 1, 0) # threshold를 기준으로 판별합니다.
        return preds

def read_cpp_code(file_path): 
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_accuracy(gt, preds): # 모델의 정확도 측정을 위한 함수입니다.
    return (gt == preds).mean()
   

val = pd.read_csv("../Dataset/sample_train.csv")
test = pd.read_csv("../Dataset/test.csv")
train_code_paths = glob.glob('../Dataset/train_code/*/*.cpp')

model = BaselineModel(threshold=0.5)

for path in tqdm(train_code_paths):
    code = read_cpp_code(path)
    model.fit([code]) # 읽어온 코드를 리스트로 묶어 모델에 학습시킵니다

val_preds = model.predict(val['code1'], val['code2'])
print(get_accuracy(val['similar'].values, val_preds))

# 모델 추론
preds = model.predict(test['code1'], test['code2'])

# 제출
submission = pd.read_csv('./sample_submission.csv')
submission['similar'] = preds
submission.to_csv('./base_submit.csv', index=False)