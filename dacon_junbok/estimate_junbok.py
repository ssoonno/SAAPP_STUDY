# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

#필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# +
#Data load
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.shape)
print(df_train)

# +
lst_x = list(df_train.columns)
lst_x.remove('id')
lst_x.remove('Target')
lst_y = ['Target']
print(lst_x)

for col_x in lst_x:
    sns.scatterplot(x= col_x, y= lst_y[0], data= df_train, hue='Gender')
    plt.show()
# -

import tensorflow as tf
from tensorflow.keras.models import Sequential 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Dense,Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# +
train_size = int(len(df_train)*0.8)
vaild_size = len(df_train) - train_size

print(train_size)
print(vaild_size)


# -

df_train[lst_x].to_numpy()[train_size:,]

# +
#Dataframe to numpy
train_size = int(len(df_train)*0.8)
vaild_size = len(df_train) - train_size

x_train = df_train[lst_x].to_numpy()[:train_size,]
y_train = df_train[lst_y].to_numpy()[:train_size,]
x_valid = df_train[lst_x].to_numpy()[train_size:,]
y_valid = df_train[lst_y].to_numpy()[train_size:,]

x_test = df_test[lst_x].to_numpy()
print('Train_x:',x_train.shape)
print('Train_y:',y_train.shape)
print('Valid_x:',x_valid.shape)
print('Valid_y:',y_valid.shape)


# +

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

EPOCHS = 1000
BATCH_SIZE = 100 #24*3


model_mlp = Sequential()
model_mlp.add(Dense(256, input_dim =x_train.shape[1]))   
model_mlp.add(Dense(512,activation=tf.keras.layers.ReLU()))
model_mlp.add(Dense(1024,activation=tf.keras.layers.ReLU()))
model_mlp.add(Dropout(0.8))
model_mlp.add(Dense(2048,activation=tf.keras.layers.ReLU()))
model_mlp.add(Dropout(0.8))
model_mlp.add(Dense(1024,activation=tf.keras.layers.ReLU()))
model_mlp.add(Dense(128,activation=tf.keras.layers.ReLU()))
model_mlp.add(Dense(1,activation=tf.keras.layers.ReLU()))
model_mlp.summary()

adam = tf.keras.optimizers.Adam()
model_mlp.compile(loss='mae', optimizer=adam)
history_cnn = model_mlp.fit(x_train,y_train,validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)

pred_y = model_mlp.predict(x_train)
print(NMAE(y_train,pred_y))

# early_stop = EarlyStopping(monitor='loss', patience=5)
# history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])


# -

pred_submit = model_mlp.predict(x_test)
pred_submit

df_submit = pd.read_csv('data/sample_submission.csv')
df_submit['Target'] =pred_submit
df_submit.to_csv('submission_1.csv',index=False)


