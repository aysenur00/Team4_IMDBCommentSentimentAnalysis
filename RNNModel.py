# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:58:34 2024

@author: deanj
"""

import pandas as pd 
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN, Dense, Activation,Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns


dataset_directory="Dataset/IMDB_Dataset.csv"
df=pd.read_csv(dataset_directory)


print(df.sentiment.value_counts(normalize = True))


# convert categorical target to numeric
def convert_target(value):
    if value=="positive":
         return 1
    else:
         return 0
     

df['sentiment']  =  df['sentiment'].apply(convert_target)

print(df[df['sentiment'] == 1].shape[0])  # pozitif sınıf örnek sayısı
print(df[df['sentiment'] == 0].shape[0])  # negatif sınıf örnek sayısı

X_train, X_test, Y_train, Y_test=train_test_split(df["review"],df["sentiment"],stratify=df["sentiment"],test_size=0.5,random_state=42)

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)

# Eğitim ve test verilerini tokenlara dönüştürme
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)

# Padding işlemi
maxlen = 120
X_train_padded = pad_sequences(X_train_tokenized, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_tokenized, maxlen=maxlen)

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = len(X_train_padded[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words,maxlen), return_sequences= False, activation="relu"))
rnn.add(Dropout(0.5))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

rnn.compile(loss = "binary_crossentropy", optimizer="rmsprop", metrics= ["accuracy"])

history = rnn.fit(X_train_padded, Y_train, validation_data= (X_test_padded,Y_test), epochs=5, batch_size=128, verbose=1)

score = rnn.evaluate(X_test_padded,Y_test)
print("Accuracy : %", score[1]* 100)


y_pred_prob = rnn.predict(X_test_padded)

y_pred = (y_pred_prob > 0.5).astype(int)

precision = precision_score(Y_test, y_pred)

recall = recall_score(Y_test, y_pred)

f1 = f1_score(Y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

labels = ['Precision', 'Recall', 'F1-Score']
values = [precision, recall, f1]

plt.bar(labels, values)
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.show()

fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
roc_auc = roc_auc_score(Y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()


