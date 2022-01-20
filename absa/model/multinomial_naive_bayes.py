#Multinomial Naive Bayes Classification:

#P(h|D) o< P(D|h) * P(h) A posterior density is (proportional to) a likelihood function times a prior distribution. The likelihood function, P(D|h) is a product.


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

#Importing Libraries for Multinomial Naive Bayes Classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


#importing pre built preprocessing library 
from preprocess import preprocess


df = pd.read_csv('C:/Users/divya/Desktop/New folder/Aspect based Sentiment detection/src/train - train.py')

df = preprocess(df)

x = df['text']
y = df['label']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


model = MultinomialNB()
model.fit(x, y)


model.score(x_test, y_test)