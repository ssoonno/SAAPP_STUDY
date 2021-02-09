# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: tf_env
#     language: python
#     name: tf_env
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

    print(dt_current.strftime('%Y-%m-%d %H:%M:%S') + ' ~ '+ (dt_current + datetime.timedelta(minutes = 200)).strftime('%Y-%m-%d %H:%M:%S') )
    dt_current = dt_current + datetime.timedelta(minutes = 200)

    if dt_current > dt_end:
      dt_current = dt_end

    querystring = {"market":str_coin,"count":200, "to":dt_current.strftime('%Y-%m-%d %H:%M:%S')}
    response = requests.request("GET", url, params=querystring)
    df_result = pd.json_normalize(response.json())
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


# df_coin = fnc_get_coinData_min("KRW-BTC","2017-12-05 00:00:00", "2021-02-01 00:00:00",10)
# df_coin = df_coin.sort_values(by='candle_date_time_utc', axis=0)
# df_coin.to_csv('df_coin.csv')


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


# +
#데이터 전처리_LSTM feature: 10분단위 가격 변화량 데이터
#가격데이터로 예측시 과거 존재하지 않았던 높은/낮은 가격일경우 에측하지 못하는 문제 발생 
#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
# print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)



df_coin = df_coin[['candle_date_time_kst','trade_price']]

df_coin['price_chage'] = df_coin['trade_price'] - df_coin['trade_price'].shift(1)

print(df_coin.head(100))

scaling_value =10000
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
plt.legend(['Train_y','Pred_y'])
plt.show()

plt.plot(Ts_real_y)
plt.plot(Ts_real_x+(pred_test_y*scaling_value))
plt.legend(['Test_y','Pred_y'])
plt.show()

# +
#데이터 전처리_CNN feature: 10분단위 6시간 가격데이터 
#Raw 데이터 불러오기
df_coin = pd.read_csv("df_coin.csv")
print(len(df_coin))
df_coin = df_coin.sort_values('candle_date_time_kst') #시간순 정렬
df_coin = df_coin.reset_index(drop = True)
print(df_coin.head(10))
print(df_coin.columns)


# df_coin = df_coin[['candle_date_time_kst','trade_price']]
# df_coin['trade_price'] = df_coin['trade_price']/100000 #data scaling
# df_coin['label'] = df_coin['trade_price'].shift(-1)
# # df_coin = df_coin.set_index('candle_date_time_kst')
# # print(df_coin.head(100))

# #lstm 학습 기준시간 6시간 (data shift 6hour ) 
# for s in range(1, 36):
#     df_coin['shift_{}'.format(s)] = df_coin['trade_price'].shift(s)
# df_coin =df_coin.dropna() #na데이터 제거 
# df_coin = df_coin[37:] #dropna가 먹히지 않아 na데이터 수동 삭제
# # print(df_coin.head(100))
# # print(df_coin.tail(100))

# #Split Train/Test Data
# max_len = len(df_coin)
# print(max_len)
# df_coin_train = df_coin.loc[:int(max_len*0.7)] #Train 70% data
# df_coin_test = df_coin[int(max_len*0.7):(max_len-1)] #Test 30% data


# df_train_x = df_coin_train.drop(['label','candle_date_time_kst'],axis = 1)
# df_train_y = df_coin_train[['label']]

# df_test_x = df_coin_test.drop(['label','candle_date_time_kst'],axis = 1)
# df_test_y = df_coin_test[['label']]
# -


