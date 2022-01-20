#LSTM : Long Short Term Memory LSTm is a model intoduced to solve the general problem of typical RNN.
#It uses specially built gates such as forget gate,input gate and output gate to solve the problem of vainishing gradients in reccurent neural networks



#Importing Basic Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string as st
import re
import string
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert (0,'C:/Users/divya/Desktop/New folder/absa/src')
#Importing Libraries for LSTM classification

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

#importing pre built preprocessing library 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from preprocess import preprocess


#Tokenizing input text 

tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

#Initiating model

model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())


# test train Split 

y=pd.get_dummies(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

#Training the model 

batch_size=32
model.fit(X_train, y_train, epochs = 5, batch_size=batch_size, verbose = 'auto')

#Evaluating the model 
model.evaluate(X_test,y_test)



