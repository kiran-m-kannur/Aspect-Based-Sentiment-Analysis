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

#Importing NLTK Libraries for preprocessing

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
wordnet_lemmatizer = WordNetLemmatizer()


# All the functions to preprocess the data:

# Lower_and_punctuation : converts all the letters into lower case and returns punctuation free string
# Tokenize : Tokenizes a given string (converts individual list)
# Remove_Small_words : removes all small words with length lesser than 3
# Remove_Stop_Words : removes all the baseic stop words in english (not required)
# Convert_list : Converts a given list to a string
# Detokenizer : Detokenizes a tokenized list
# Find_aspect : Finds Aspect in the given list of clauses
# conj_based_split : Splits a given list into respective clauses by breaking conjunctions


def lower_and_punctuation(data):
  punctuationfree="".join([i for i in data if i not in string.punctuation])
  return punctuationfree


def find_aspect(df):
  result=[]
  for index in range(len(df)):
    res = [i for i in df['text'][index] if df['aspect'][index] in i]
    result.append(res)
  df['text'] = result
  return df


def conj_based_split(text):
  conj =['and','but','however']
  punctuation_lst =['.','!',',',':',';','?']
  conj= conj+punctuation_lst
  lst=[]
  lst2 = []
  a=1
  for word in text.split():
    if word not in conj:
      lst.append(word)
    elif (word in conj):
      sent=' '.join(lst)
      lst2.append(sent)
      lst.clear()
    if (a==len(text.split())):
      if(len(lst)):
        sent=' '.join(lst)
        lst2.append(sent)
    a=a+1
  return lst2

  
def convertList(list1):  
  str = ''
  for i in list1: 
      str += i  
  return str 

def tokenize(text):
  text = re.split('\s+' ,text)
  return [x.lower() for x in text]


def remove_small_words(text):
  return [x for x in text if len(x) > 3 ]


def remove_stopwords(text):
  output= [i for i in text if i not in stopwords]
  return output


def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text


def detokenizer(text):
  str1 = " " 
  return (str1.join(text))

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def preprocess(df):
  df['aspect']= df['aspect'].apply(lambda x: x.lower())
  df['text']= df['text'].apply(lambda x: x.lower())
  df['text']= df['text'].apply(lambda x:remove_punctuation(x))
  df['aspect']= df['aspect'].apply(lambda x:remove_punctuation(x))
  df['text']= df['text'].apply(lambda x:conj_based_split(x))
  df = find_aspect(df)
  df['text']= df['text'].apply(lambda x:convertList(x))
  #df.drop(df.loc[df['label']==1].index, inplace=True)

  return df

  