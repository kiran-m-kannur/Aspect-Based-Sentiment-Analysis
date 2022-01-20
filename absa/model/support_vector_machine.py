#Support Vector Machine

#SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.


#Importing Basic Libraries

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
sys.path.insert (0,'C:/Users/divya/Desktop/New folder/absa/src')
#Importing Libraries for Support Vector Machine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline, Pipeline

#importing pre built preprocessing library 
from preprocess import preprocess

df  = pd.read_csv()

preprocess(df)

train, test = train_test_split(df,test_size=0.2, random_state=1)
X_train = train['text'].values
X_test = test['text'].values
y_train = train['label']
y_test = test['label']


#Preprocessing for the training

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

print(grid_svm.best_score_)
