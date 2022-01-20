#Evaluation Time !!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string as st
import re
import string
import os
import warnings
warnings.filterwarnings("ignore")

# imports preprocess from preprocess.py file
from preprocess import preprocess


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline, Pipeline



#importing test dataframe
test_df = pd.read_csv('C:/Users/divya/Desktop/New folder/Aspect based Sentiment detection/src')

#preprocessing the data
preprocess(test_df)

#PERFORMING SVM 

train, test = train_test_split(test_df,test_size=0.2, random_state=1)
X_train = train['text'].values
X_test = test['text'].values
y_train = train['label']
y_test = test['label']


def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

en_stopwords = set(stopwords.words("english")) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)

#Defining the model
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
np.random.seed(1)
pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)


def predict(df):
  test_df = preprocess(df)
  preds=[]
  for i in range(len(test_df)):
  #print(test_df['text'][i])
    pred=int(grid_svm.predict([test_df['text'][i]]))
    preds.append(pred)
  test_df['label']=np.array(preds)

predict(test_df)

df.to_csv('C:/Users/divya/Desktop/New folder/absa/results/evaluation.py')