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

# # Coin Trading
#
#

# 필요 라이브러리 import
import pyupbit
import urllib.request
import json
import codecs
import time
import requests
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#API 테스트
#원화로 거래되는 코인종류 확인
tickers = pyupbit.get_tickers(fiat="KRW")
print(tickers)  
#현재 거래되는 코인가격확인 (ex. BTC-XRP XRP(리플)의 비트코인 가격)
price = pyupbit.get_current_price(["KRW-BTC", "KRW-XRP"])
print(price)
#과거 이력 조회
df = pyupbit.get_ohlcv("KRW-BTC")
# print(df)
df_min = pyupbit.get_ohlcv(ticker = 'KRW-BTC',interval='minute1',count =1000)
# print(len(df_min))

# ## 필요 함수 생성

# +
#데이터 수집 API호출
def fnc_get_coinData_min(str_coin, str_start, str_end, unit):
    

    url = "https://api.upbit.com/v1/candles/minutes/" + str(unit)
    df_coin = pd.DataFrame()

    #querystring = {"market":"KRW-BTC","count":"200","to":str_endTime}
    dt_start = datetime.datetime.strptime(str_start, '%Y-%m-%d %H:%M:%S')
    dt_end = datetime.datetime.strptime(str_end, '%Y-%m-%d %H:%M:%S')
    dt_current = dt_start
    while dt_current < dt_end:
        print(dt_current.strftime('%Y-%m-%d %H:%M:%S')+' ~ '+(dt_current + datetime.timedelta(minutes = 200)).strftime('%Y-%m-%d %H:%M:%S'))
        dt_current = dt_current + datetime.timedelta(minutes = 200)

        if dt_current > dt_end:
            dt_current = dt_end

        querystring = {"market":str_coin,"count":200, "to":dt_current.strftime('%Y-%m-%d %H:%M:%S')}
        response = requests.request("GET", url, params=querystring)
        try:
            df_result = pd.json_normalize(response.json())
        except:
            print("API Connection Error")
            dt_current = dt_current - datetime.timedelta(minutes = 200)
            continue
                
        df_coin = pd.concat([df_coin,df_result],ignore_index=True)
        print(len(df_coin))
  
    return df_coin

def fnc_plot_mResult(history):
    acc = history.history['accuracy']
    val_acc =history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']


    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Acc', 'Val_Acc'])
    plt.show()


    plt.plot(loss)
    plt.plot(val_loss)

    plt.legend(['Loss', 'Val_Loss'])
    plt.show()


# df_coin = fnc_get_coinData_min("KRW-BTC","2017-12-07 00:00:00", "2021-02-01 00:00:00",10)
# df_coin = df_coin.sort_values(by='candle_date_time_utc', axis=0)
# df_coin.to_csv('df_coin.csv')


# -


# ## 1.LSTM with 6hour price pattern(using only price data)
# #### 1. 사용 feature : 과거 6시간의 10분단위 가격데이터(35 컬럼)
# #### 2. 로직 : 6시간의 가격데이터 패턴으로 10분뒤 가격 예측

df_coin = pd.read_csv("df_coin.csv")
# print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)


# +
#데이터 전처리_LSTM feature: 10분단위 6시간 가격데이터 
#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
# print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)



df_coin = df_coin[['candle_date_time_kst','trade_price']]
scaling_value =1000000
df_coin['trade_price'] = df_coin['trade_price']/scaling_value #data scaling
df_coin['label'] = df_coin['trade_price'].shift(-1)
# df_coin = df_coin.set_index('candle_date_time_kst')
print("Data head: ")
print(df_coin.head(100))

#lstm 학습 기준시간 6시간 (data shift 6hour ) 
for s in range(1, 36):
    df_coin['shift_{}'.format(s)] = df_coin['trade_price'].shift(s)
df_coin =df_coin.dropna() #na데이터 제거 
df_coin = df_coin[37:] #dropna가 먹히지 않아 na데이터 수동 삭제
# print(df_coin.head(100))
# print(df_coin.tail(100))

#Split Train/Test Data
max_len = len(df_coin)
print("total data rows: "+ str(max_len) )
df_coin_train = df_coin.loc[:int(max_len*0.7)] #Train 70% data
df_coin_test = df_coin[int(max_len*0.7):(max_len-1)] #Test 30% data
print("Train data rows: "+ str(len(df_coin_train)))
print("Test data rows: "+ str(len(df_coin_test)))

df_train_x = df_coin_train.drop(['label','candle_date_time_kst'],axis = 1)
df_train_y = df_coin_train[['label']]

df_test_x = df_coin_test.drop(['label','candle_date_time_kst'],axis = 1)
df_test_y = df_coin_test[['label']]


# +
#modeling_LSTM feature: 10분단위 6시간 가격데이터 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow.keras.backend as K 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

EPOCHS = 100
BATCH_SIZE = 500

#모델에 사용하기위해 dataframe -> numpy 형식으로 변환
Train_x = df_train_x.values
Train_y = df_train_y.values
Test_x = df_test_x.values
Test_y = df_test_y.values

print(Train_x.shape)
print(Train_y.shape)
print(Test_x.shape)
print(Test_y.shape)
print(Train_x)

#LSTM 데이터 형식으로 변환
Train_x = Train_x.reshape(Train_x.shape[0],36,1)
Test_x = Test_x.reshape(Test_x.shape[0],36,1)
print(Train_x.shape)
print(Test_x.shape)

#모델 생성
K.clear_session()
model_lstm = Sequential() # Sequeatial Model
model_lstm.add(LSTM(40, input_shape=(36, 1))) # (timestep, feature)
model_lstm.add(Dense(1)) # output = 1
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model_lstm.summary()

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

history = model_lstm.fit(Train_x, Train_y, validation_data = (Test_x,Test_y)
                         ,epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop])

fnc_plot_mResult(history)

#모델 저장
# model_lstm.save('model_lstm')
# print("Model save complete!!")

# +
#모델 로드
#reconstructed_model = tf.keras.models.load_model("my_model")
#예측 결과 확인
pred_train_y = model_lstm.predict(Train_x)
pred_test_y = model_lstm.predict(Test_x)
# print(Train_y)
# print(pred_train_y)

plt.plot(Train_y*scaling_value)
plt.plot(pred_train_y*scaling_value)
plt.legend(['Train_y','Pred_y'])
plt.show()

plt.plot(Test_y*scaling_value)
plt.plot(pred_test_y*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.show()
# -


# ## 2.LSTM with 6hour price change pattern(using only price change data)
# #### 가격데이터만으로 예측을 진행할 경우 과거 학습하지 못한 더높은 가격의 경우 예측하지 못하는 현상 발생
# #### 1. 사용 feature : 과거 6시간의 10분단위 가격 변동 데이터(35 컬럼)
# #### 2. 로직 : 6시간의 가격변동 데이터 패턴으로 10분뒤 가격의 변동성 예측
# #### -> 변동성 데이터는 0인 라벨을 줄이고 시계열 모델이 아닌 다른 모델로 접근해보자
#

# +
#데이터 전처리_LSTM feature: 10분단위 가격 변화량 데이터
#가격데이터로 예측시 과거 존재하지 않았던 높은/낮은 가격일경우 에측하지 못하는 문제 발생하여 가격 변화량 데이터 사용
#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
# print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)



df_coin = df_coin[['candle_date_time_kst','trade_price']]

df_coin['price_chage'] = df_coin['trade_price'] - df_coin['trade_price'].shift(1)

print(df_coin.head(100))

scaling_value =100
df_coin['price_chage'] = df_coin['price_chage']/scaling_value #data scaling
df_coin['label'] = df_coin['price_chage'].shift(-1)
# df_coin = df_coin.set_index('candle_date_time_kst')


#lstm 학습 기준시간 6시간 (data shift 6hour ) 
for s in range(1, 36):
    df_coin['shift_{}'.format(s)] = df_coin['price_chage'].shift(s)
df_coin =df_coin.dropna() #na데이터 제거 
df_coin = df_coin[38:] #dropna가 먹히지 않아 na데이터 수동 삭제
print("Data head: ")
print(df_coin.head(100))

#Split Train/Test Data
max_len = len(df_coin)
print("total data rows: "+ str(max_len) )
df_coin_train = df_coin[:int(max_len*0.7)] #Train 70% data
df_coin_test = df_coin[int(max_len*0.7):(max_len-1)] #Test 30% data
print("Train data rows: "+ str(len(df_coin_train)))
print("Test data rows: "+ str(len(df_coin_test)))

df_coin_train.head()
df_train_x = df_coin_train.drop(['label','candle_date_time_kst','trade_price'],axis = 1)
df_train_y = df_coin_train[['label']]
df_tr_real_x = df_coin_train[['trade_price']]
df_tr_real_y = df_coin_train[['trade_price']].shift(-1)

df_test_x = df_coin_test.drop(['label','candle_date_time_kst','trade_price'],axis = 1)
df_test_y = df_coin_test[['label']]
df_ts_real_x = df_coin_test[['trade_price']]
df_ts_real_y = df_coin_test[['trade_price']].shift(-1)




# +
#modeling_LSTM feature: 10분단위 6시간 가격데이터 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow.keras.backend as K 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

EPOCHS = 100
BATCH_SIZE = 500

#모델에 사용하기위해 dataframe -> numpy 형식으로 변환
Train_x = df_train_x.values
Train_y = df_train_y.values
Test_x = df_test_x.values
Test_y = df_test_y.values

print(Train_x.shape)
print(Train_y.shape)
print(Test_x.shape)
print(Test_y.shape)

#LSTM 데이터 형식으로 변환
Train_x = Train_x.reshape(Train_x.shape[0],36,1)
Test_x = Test_x.reshape(Test_x.shape[0],36,1)
print(Train_x.shape)
print(Test_x.shape)

#모델 생성
K.clear_session()
model_lstm = Sequential() # Sequeatial Model
model_lstm.add(LSTM(36, input_shape=(36, 1))) # (timestep, feature)
# model_lstm.add(Dense(1,activation='relu')) # output = 1
model_lstm.add(Dense(1)) # output = 1
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model_lstm.summary()

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

history = model_lstm.fit(Train_x, Train_y, validation_data = (Test_x,Test_y)
                         ,epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop])

fnc_plot_mResult(history)

#모델 저장
# model_lstm.save('model_lstm')
# print("Model save complete!!")

# +
#모델 로드
#reconstructed_model = tf.keras.models.load_model("my_model")
#예측 결과 확인


#모델결과 확인을 위해 실제 price데이터 dataframe -> numpy 형식으로 변환
Tr_real_x = df_tr_real_x.values
Tr_real_y = df_tr_real_y.values
Ts_real_x = df_ts_real_x.values
Ts_real_y = df_ts_real_y.values


pred_train_y = model_lstm.predict(Train_x)
pred_test_y = model_lstm.predict(Test_x)
# print(Train_y)
# print(pred_train_y)

plt.plot(Tr_real_y)
plt.plot(Tr_real_x+(pred_train_y*scaling_value))
plt.legend(['Train_real_y','Pred_real_y'])
plt.show()

plt.plot(Ts_real_y)
plt.plot(Ts_real_x+(pred_test_y*scaling_value))
plt.legend(['Test_real_y','Pred_real_y'])
plt.show()


plt.plot(Train_y*scaling_value)
plt.plot(pred_train_y*scaling_value)
plt.legend(['Train_y','Pred_y'])
plt.show()

plt.plot(Test_y*scaling_value)
plt.plot(pred_test_y*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.show()
# -

# ## 3.CNN 

# +
#데이터 전처리_CNN feature: 10분단위 6시간 가격데이터 

#단위 수정
scaling_value =10000

#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)
print(df_coin.head(10))
# print(df_coin.columns)
df_coin[['opening_price', 'high_price', 'low_price', 'trade_price','candle_acc_trade_price']] = df_coin[['opening_price', 'high_price', 'low_price', 'trade_price','candle_acc_trade_price']]/scaling_value

df_coin = df_coin[['opening_price', 'high_price', 'low_price', 'trade_price','candle_acc_trade_price', 'candle_acc_trade_volume']]
# df_y = df_coin[['trade_price']].shift(-1)
df_y = df_coin[['trade_price']]
lst_result = []
lst_y1 = []
lst_y2 = []
for i in range(35,len(df_coin)-6):

    if  round(100*((i+1)/len(df_coin)))>round(100*((i)/len(df_coin))):
        print("Process :"+str( round(100*((i+1)/len(df_coin)),2) )+"("+str(i+1)+"/"+str(len(df_coin))+")") #진행상태 확인
        
    df_tmp = df_coin.iloc[i-35:i-1]
    lst_result.append(df_tmp.transpose().values)
#     lst_y1.append(df_y.transpose().iloc[i:i+35])
    lst_y1.append(df_y.iloc[i:i+6].transpose().values)
    lst_y2.append(df_y.iloc[i])


print("PREPROCESS END")
x_data = np.array(lst_result)
y_data1 = np.array(lst_y1)
y_data2 = np.array(lst_y2)
print(x_data.shape)
print(y_data1.shape) # 다음 1시간(6 step)예측
print(y_data2.shape) # 10분뒤(1step) 예측

#전처리 데이터 저장
np.save('x_cnn_data',x_data)
np.save('y_cnn_data1',y_data1)
np.save('y_cnn_data2',y_data2)
# -
# ### 3-1 CNN 10분뒤 예측 (추세는 예측하나 상승, 하강 시점을 못잡음)

# +
#단위 수정
scaling_value =10000

#Data 로드
x_data = np.load('x_cnn_data.npy')
y_data1 = np.load('y_cnn_data1.npy')
y_data2= np.load('y_cnn_data2.npy')

#Train/Test 분리
max_len = len(x_data)
df_coin_train = x_data[:int(max_len*0.7)] #Train 70% data
df_coin_test = x_data[int(max_len*0.7):(max_len-1)] #Test 30% data

df_result_train_1h = y_data1[:int(max_len*0.7)] #Train 70% data
df_result_test_1h = y_data1[int(max_len*0.7):(max_len-1)] #Test 30% data
df_result_train = y_data2[:int(max_len*0.7)] #Train 70% data
df_result_test = y_data2[int(max_len*0.7):(max_len-1)] #Test 30% data

print(df_coin_train.shape)
print(df_coin_test.shape)
print(df_result_train.shape)
print(df_result_test.shape)
print(df_result_train_1h.shape)
print(df_result_test_1h.shape)


#modeling_CNN  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Dense, Conv2D,Conv1D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

EPOCHS = 100
BATCH_SIZE = 64

dt_rows = df_coin_train.shape[1] #feature종류
dt_cols = df_coin_train.shape[2] #timestep

input_shape = (dt_rows, dt_cols, 1)







# -


# ## CNN_10min Result

# +
#Conv2D
# x_train = df_coin_train.reshape(df_coin_train.shape[0], dt_rows, dt_cols, 1)
# x_test = df_coin_test.reshape(df_coin_test.shape[0], dt_rows, dt_cols, 1)
# y_train = df_result_train.reshape(df_result_train.shape[0],1,dt_cols)
# y_test = df_result_test.reshape(df_result_test.shape[0],1,dt_cols)

#Conv1D
x_train = df_coin_train.reshape(df_coin_train.shape[0], dt_rows, dt_cols)
x_test = df_coin_test.reshape(df_coin_test.shape[0], dt_rows, dt_cols)
y_train = df_result_train
y_test = df_result_test
y_train_1h = df_result_train_1h.reshape(df_result_train_1h.shape[0],df_result_train_1h.shape[2])
y_test_1h = df_result_test_1h.reshape(df_result_test_1h.shape[0],df_result_test_1h.shape[2])

# model_cnn = Sequential()
# model_cnn.add(Conv2D(32,  kernel_size=(3,3), input_shape =(dt_rows, dt_cols, 1),activation='relu'))
# # model_cnn.add(MaxPooling2D(pool_size=(2,2)))
# model_cnn.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
# model_cnn.add(MaxPooling2D(pool_size=(2,2)))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(512,activation='relu'))
# model_cnn.add(Dense(128,activation='relu'))
# model_cnn.add(Dense(20,activation='relu'))
# model_cnn.add(Dropout(0.5))
# model_cnn.add(Dense(1))

#10min forecast
model_cnn = Sequential()
model_cnn.add(Conv1D(32,  kernel_size=1, input_shape =(dt_rows, dt_cols),activation='relu'))
model_cnn.add(Conv1D(64,  kernel_size=dt_rows, activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(512,activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(1))


model_cnn.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
tf.keras.optimizers.Adam(learning_rate=1e-5)
history_cnn = model_cnn.fit(x_train,y_train,validation_data =(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1, callbacks=[early_stop])

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()


#모델 로드
#reconstructed_model = tf.keras.models.load_model("my_model")
#예측 결과 확인
# pred_train_y = model_cnn.predict(x_train)
pred_test_y = model_cnn.predict(x_test)
print("Predict Complete!")

plt.plot(y_test*scaling_value)
plt.plot(pred_test_y*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('10min forecast_test')
plt.show()


plt.plot(y_test[29112:30627]*scaling_value)
plt.plot(pred_test_y[29112:30627]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_1')
plt.show()


plt.plot(y_test[432600:433537]*scaling_value)
plt.plot(pred_test_y[432600:433537]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_2')
plt.show()





# -

# ## CNN 결과

# +
#Conv2D
# x_train = df_coin_train.reshape(df_coin_train.shape[0], dt_rows, dt_cols, 1)
# x_test = df_coin_test.reshape(df_coin_test.shape[0], dt_rows, dt_cols, 1)
# y_train = df_result_train.reshape(df_result_train.shape[0],1,dt_cols)
# y_test = df_result_test.reshape(df_result_test.shape[0],1,dt_cols)

#Conv1D
x_train = df_coin_train.reshape(df_coin_train.shape[0], dt_rows, dt_cols)
x_test = df_coin_test.reshape(df_coin_test.shape[0], dt_rows, dt_cols)
y_train = df_result_train
y_test = df_result_test
y_train_1h = df_result_train_1h.reshape(df_result_train_1h.shape[0],df_result_train_1h.shape[2])
y_test_1h = df_result_test_1h.reshape(df_result_test_1h.shape[0],df_result_test_1h.shape[2])

# model_cnn = Sequential()
# model_cnn.add(Conv2D(32,  kernel_size=(3,3), input_shape =(dt_rows, dt_cols, 1),activation='relu'))
# # model_cnn.add(MaxPooling2D(pool_size=(2,2)))
# model_cnn.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
# model_cnn.add(MaxPooling2D(pool_size=(2,2)))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(512,activation='relu'))
# model_cnn.add(Dense(128,activation='relu'))
# model_cnn.add(Dense(20,activation='relu'))
# model_cnn.add(Dropout(0.5))
# model_cnn.add(Dense(1))

#10min forecast
model_cnn = Sequential()
model_cnn.add(Conv1D(32,  kernel_size=1, input_shape =(dt_rows, dt_cols),activation='relu'))
model_cnn.add(Conv1D(64,  kernel_size=dt_rows, activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(512,activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(1))


#1h forecast
# model_cnn_1h = Sequential()
# model_cnn_1h.add(Conv1D(32,  kernel_size=1, input_shape =(dt_rows, dt_cols),activation='relu'))
# model_cnn_1h.add(Conv1D(64,  kernel_size=dt_rows,activation='relu'))
# model_cnn_1h.add(Flatten())
# model_cnn_1h.add(Dense(512,activation='relu'))
# model_cnn_1h.add(Dropout(0.2))
# model_cnn_1h.add(Dense(6))

# print(model_cnn_1h.summary())
# model_cnn_1h.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
# early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
# history_cnn_1h = model_cnn_1h.fit(x_train,y_train_1h,validation_data =(x_test, y_test_1h), batch_size=BATCH_SIZE, epochs=EPOCHS,callbacks= early_stop)

# plt.plot(history_cnn_1h.history['loss'])
# plt.plot(history_cnn_1h.history['val_loss'])
# plt.legend(['loss','val_loss'])
# plt.show()


model_cnn.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
# model_cnn.compile(optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
tf.keras.optimizers.Adam(learning_rate=1e-5)
history_cnn = model_cnn.fit(x_train,y_train,validation_data =(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1, callbacks=[early_stop])
# fnc_plot_mResult(history_cnn)

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()





#모델 로드
#reconstructed_model = tf.keras.models.load_model("my_model")
#예측 결과 확인
# pred_train_y = model_cnn.predict(x_train)
pred_test_y = model_cnn.predict(x_test)
# pred_train_y = model_cnn_1h.predict(x_train)
pred_test_y_1h = model_cnn_1h.predict(x_test)
print("Predict Complete!")

# plt.plot(y_train)
# plt.plot(pred_train_y)
# plt.legend(['Train_y','Pred_y'])
# plt.title('10min forecast_train')
# plt.show()

plt.plot(y_test*scaling_value)
plt.plot(pred_test_y*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('10min forecast_test')
plt.show()


plt.plot(y_test[29112:30627]*scaling_value)
plt.plot(pred_test_y[29112:30627]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_1')
plt.show()


plt.plot(y_test[432600:433537]*scaling_value)
plt.plot(pred_test_y[432600:433537]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_2')
plt.show()



plt.plot(y_test_1h*scaling_value)
plt.plot(pred_test_y_1h*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('1h forecast_test')
plt.show()


plt.plot(y_test_1h[29112:30627]*scaling_value)
plt.plot(pred_test_y_1h[29112:30627]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_1')
plt.show()


plt.plot(y_test_1h[432600:433537]*scaling_value)
plt.plot(pred_test_y_1h[432600:433537]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_2')
plt.show()



# -

# ## CNN + LSTM 결과

# +
#CNN + LSTM
model_cnn = Sequential()
model_cnn.add(Conv1D(32,  kernel_size=3, input_shape =(dt_rows, dt_cols),activation='relu'))
# model_cnn.add(Conv1D(64,  kernel_size=3,activation='relu'))
model_cnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model_cnn.add(Dense(512,activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(128,activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(1))
print(model_cnn.summary())

model_cnn.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
tf.keras.optimizers.Adam(learning_rate=1e-5)
history_cnn = model_cnn.fit(x_train,y_train,validation_data =(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1, callbacks=[early_stop])
# plt.semilogx(history_cnn.history["lr"], history_cnn.history["loss"])


plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()

#예측 결과 확인
# pred_train_y = model_cnn.predict(x_train)
pred_test_y = model_cnn.predict(x_test)
print("Predict Complete!")

plt.plot(y_test*scaling_value)
plt.plot(pred_test_y*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('10min forecast_test')
plt.show()


plt.plot(y_test[29112:30627]*scaling_value)
plt.plot(pred_test_y[29112:30627]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_1')
plt.show()


plt.plot(y_test[432600:433537]*scaling_value)
plt.plot(pred_test_y[432600:433537]*scaling_value)
plt.legend(['Test_y','Pred_y'])
plt.title('Problem_point_2')
plt.show()


# -

# ### CNN 변화량 예측

print(y_test_1h.tolist()[0])
print(pred_test_y_1h.tolist()[0])

# +
#데이터 전처리_CNN feature: 10분단위 6시간 가격데이터 

#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)
print(df_coin.head(10))
# print(df_coin.columns)


df_coin = df_coin[['opening_price', 'high_price', 'low_price', 'trade_price','candle_acc_trade_price', 'candle_acc_trade_volume']]
# df_y = df_coin[['trade_price']].shift(-1)
df_y = df_coin[['trade_price']]
lst_result = []
lst_y1 = []
lst_y2 = []
for i in range(35,len(df_coin)-6):

    if  round(100*((i+1)/len(df_coin)))>round(100*((i)/len(df_coin))):
        print("Process :"+str( round(100*((i+1)/len(df_coin)),2) )+"("+str(i+1)+"/"+str(len(df_coin))+")") #진행상태 확인
        
    df_tmp = df_coin.iloc[i-35:i-1]
    lst_result.append(df_tmp.transpose().values)
#     lst_y1.append(df_y.transpose().iloc[i:i+35])
    lst_y1.append(df_y.iloc[i:i+6].transpose().values)
    lst_y2.append(df_y.iloc[i])


print("PREPROCESS END")
x_data = np.array(lst_result)
y_data1 = np.array(lst_y1)
y_data2 = np.array(lst_y2)
print(x_data.shape)
print(y_data1.shape) # 다음 6시간(35 step)예측
print(y_data2.shape) # 10분뒤(1step) 예측

#전처리 데이터 저장
np.save('x_cnn_data',x_data)
np.save('y_cnn_data1',y_data1)
np.save('y_cnn_data2',y_data2)
# -


