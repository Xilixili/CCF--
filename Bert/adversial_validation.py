'''
adversial validation:
this is to make sure that the training data and the test data come from the same distribution
-if the clf's result is great, the competition might not be suitable for you
'''
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
rows = 10000
train_df = pd.read_csv('./dataset/unlabeled_data.csv', nrows=rows)
test_df = pd.read_csv('./dataset/test_data.csv', nrows=rows)
print(train_df['content'])
# 转成了array
print(train_df['content'].iloc[:].values)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500).fit(train_df['content'].iloc[:].values.astype('U'))
train_tfidf = tfidf.transform(train_df['content'].iloc[:].values.astype('U'))
test_tfidf = tfidf.transform(test_df['content'].iloc[:].values.astype('U'))

train_test = np.vstack([train_tfidf.toarray(), test_tfidf.toarray()])  # new training data
lgb_data = lgb.Dataset(train_test, label=np.array([1] * rows + [0] * rows))
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['metric'] = 'auc'
result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=3, verbose_eval=20)
print(pd.DataFrame(result))
