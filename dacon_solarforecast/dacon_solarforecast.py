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

# +
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import pytz
from sklearn.linear_model import LinearRegression
from tensorflow.keras import backend as K
import tensorflow as tf
def fnc_makeDate(date, hour):
    retrun_date = date
    if hour >= 24 :
        retrun_date = date + timedelta(days=1)
    return retrun_date

def fnc_makeHour( hour):
    return_hour = hour
    if hour >= 24 :
        return_hour = return_hour - 24
    return int(return_hour)


def sola_nmae(answer_df, submission_df):
    # submission = submission_df[submission_df['time'].isin(answer_df['time'])]
    # submission.index = range(submission.shape[0])
    
    # 시간대별 총 발전량
    sum_submission = submission_df.sum(axis=1)
    sum_answer = answer_df.sum(axis=1)
    
    # 발전소 발전용량
    capacity = {
        'dangjin_floating':1000, # 당진수상태양광 발전용량
        'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
        'dangjin':1000, # 당진태양광 발전용량
        'ulsan':500 # 울산태양광 발전용량
    }
    
    # 총 발전용량
    total_capacity = np.sum(list(capacity.values()))
    
    # 총 발전용량 절대오차
    absolute_error = (sum_answer - sum_submission).abs()
    
    # 발전용량으로 정규화
    absolute_error /= total_capacity
    
    # 총 발전용량의 10% 이상 발전한 데이터 인덱스 추출
    target_idx = sum_answer[sum_answer>=total_capacity*0.1].index
    
    # NMAE(%)
    nmae = 100 * absolute_error[target_idx].mean()
    
    return nmae

def nmae_10(y_true, y_pred):
    

    absolute_error = K.abs(y_true - y_pred)
    absolute_error /= capacity
    print(y_pred)
    print((K.greater(y_pred,capacity*0.1)))
#     y_pred = K.greater(y_pred,capacity*0.1)
#     y_pred = K.cast(out,K.floatx())
    
#     target_idx = np.where(y_true>=capacity*0.1)
    
    loss = 100 * K.mean(absolute_error)
    
    
    
    return loss
# -

# # 발전량 1개 씩 24개 예측

# +
df_data = pd.read_csv('data/df_data.csv')
df_real_test = pd.read_csv('data/df_real_test.csv')
lst_site = df_data['variable'].unique()
df_data_sum = df_data[['variable','location','fcst_date','value']].groupby(['variable','location','fcst_date'], as_index=False).sum()
df_data_sum.columns = ['variable','location','fcst_date','val_day']
df_data = pd.merge(df_data,df_data_sum)


# lst_site = ['dangjin']
df_submit = pd.DataFrame()
df_answer = pd.DataFrame()
df_submit_valid = pd.DataFrame()
df_answer_valid = pd.DataFrame()
df_real_submit = pd.DataFrame()
# df_submit['time'] = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-31 23:00:00', freq='H')
# df_answer['time'] = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-31 23:00:00', freq='H')
df_real_submit['time'] = pd.date_range(start='2021-02-01 00:00:00', end='2021-02-28 23:00:00', freq='H') #제출용
lst_all_features = ['fcst_date','fcst_hour','altitude','radiation','Temperature','Humidity','WindSpeed','WindDirection','Cloud','rain','rain_6h','rain_prob','snow_6h']
lst_features = ['fcst_hour','altitude','radiation','WindSpeed','Temperature','Humidity','Cloud','rain','rain_prob','rain_6h','snow_6h']

mx_alt = 90
mx_angle = 365
mx_rad = 1000
mx_temp = 35
mx_hum = 100
mx_windsp = 34
mx_cloud =4
mx_rain = 2
mx_rain_6 =85
mx_snow_6 =2.5
mx_rain_prob = 90
CP = {
    'dangjin_floating':1000, # 당진수상태양광 발전용량
    'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
    'dangjin':1000, # 당진태양광 발전용량
    'ulsan':500 # 울산태양광 발전용량
}
for site in lst_site:

    capacity =CP[site]
    print(site,'/',capacity,'#'*60)
    df_data_site = df_data[df_data['variable']==site].copy()
    df_data_site = df_data_site[ (df_data_site['fcst_date']<'2021-02-01') & (df_data_site['fcst_date']!='2018-03-17') & (df_data_site['fcst_date']!='2018-03-18') & (df_data_site['fcst_date']!='2018-03-19') & (df_data_site['fcst_date']!='2020-06-26')&(df_data_site['fcst_date']!='2020-06-27')  ]
 
#     mx_val = df_data_site['value'].max()
#     print('max_val :',mx_val)
    
    df_data_site= df_data_site[lst_all_features+['value']]
    df_data_site = df_data_site.sort_values(['fcst_date','fcst_hour'], axis = 0)    
    df_data_site['altitude'] = df_data_site['altitude']/mx_alt
#     df_data_site['angle'] = df_data_site['angle']/mx_angle
    df_data_site['radiation'] = df_data_site['radiation']/mx_rad
    df_data_site['Temperature'] = df_data_site['Temperature']/mx_temp
    df_data_site['Humidity'] = df_data_site['Humidity']/mx_hum
    df_data_site['WindSpeed'] = df_data_site['WindSpeed']/mx_windsp
    df_data_site['Cloud'] = df_data_site['Cloud']/mx_cloud
    df_data_site['fcst_hour'] = df_data_site['fcst_hour']/24
    df_data_site['rain'] = df_data_site['rain']/mx_rain
    df_data_site['rain_6h'] = df_data_site['rain_6h']/mx_rain_6
    df_data_site['rain_prob'] = df_data_site['rain_prob']/mx_rain_prob
    df_data_site['snow_6h'] = df_data_site['snow_6h']/mx_snow_6
#     df_data_site['value'] = df_data_site['value']/mx_val

    
    df_train = df_data_site[df_data_site['fcst_date']<'2021-01-01'].copy()
    print(len(df_train))
#     df_train = pd.concat([df_train[df_train['value']>0],df_train[df_train['value']==0].sample(frac=0.4,random_state=1004)],axis = 0)
    df_train = df_train[df_train['value']>0]
    print(len(df_train))
    df_test = df_data_site[(df_data_site['fcst_date']>='2021-01-01')&(df_data_site['fcst_date']<'2021-02-01')].copy()
    df_test = df_test.sort_values(['fcst_date','fcst_hour'], axis = 0)
    
    #제출용 데이터
    df_real = df_real_test[df_real_test['variable']==site].copy()
    df_real['altitude'] = df_real['altitude']/mx_alt
#     df_real['angle'] = df_real['angle']/mx_angle
    df_real['radiation'] = df_real['radiation']/mx_rad
    df_real['Temperature'] = df_real['Temperature']/mx_temp
    df_real['Humidity'] = df_real['Humidity']/mx_hum
    df_real['WindSpeed'] = df_real['WindSpeed']/mx_windsp
    df_real['Cloud'] = df_real['Cloud']/mx_cloud
    df_real['fcst_hour'] = df_real['fcst_hour']/24
    df_real['rain'] = df_real['rain']/mx_rain
    df_real['rain_6h'] = df_real['rain_6h']/mx_rain_6
    df_real['rain_prob'] = df_real['rain_prob']/mx_rain_prob
    df_real['snow_6h'] = df_real['snow_6h']/mx_snow_6

    df_real= df_real[lst_all_features]
    df_real = df_real.sort_values(['fcst_date','fcst_hour'], axis = 0)    
    
    
    x_train = np.array(df_train[lst_features].values)
    x_valid = np.array(df_test[lst_features].values)
    x_submit = np.array(df_data_site[lst_features].values)
    x_test = np.array(df_real[lst_features].values)
#     y_train = np.array(df_train['value'].values).astype('int64')
#     y_valid = np.array(df_test['value'].values).astype('int64')
#     y_submit = np.array(df_data_site['value'].values.astype('int64'))
    y_train = np.array(df_train['value'].values).astype('float64')
    y_valid = np.array(df_test['value'].values).astype('float64')
    y_submit = np.array(df_data_site['value'].values.astype('float64'))
    print('Train :', x_train.shape, '/',y_train.shape)
    print('Vaild :', x_valid.shape, '/',y_valid.shape)
    
#     print(df_train[lst_features+['value']].head(24))
    
    #모델 생성
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    import tensorflow.keras.backend as K 
    from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D, Dropout, Flatten, LSTM
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt

    EPOCHS = 1000
    BATCH_SIZE = 128 #24*3

    dt_rows = x_train.shape[1] #feature종류

#     model_cnn = Sequential()
#     model_cnn.add(Dense(128, input_dim =dt_rows))  
#     model_cnn.add(Dense(256, activation='relu'))  
# #     model_cnn.add(Dense(2048,activation='relu'))
# #     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1))
#     model_cnn.summary()


    model_cnn = Sequential()
    model_cnn.add(Dense(256, input_dim =dt_rows))   
    model_cnn.add(Dense(512,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(1024,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(256,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(1,activation=tf.keras.layers.ReLU()))
    model_cnn.summary()

#     model_cnn = Sequential()
#     model_cnn.add(Dense(256, input_dim =dt_rows,activation = 'relu'))  
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dense(128,activation='relu'))
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dense(256,activation='relu'))
#     model_cnn.add(Dense(128,activation='relu'))
#     model_cnn.add(Dense(256,activation='relu'))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dense(1,activation='relu'))
#     model_cnn.summary()
    
    
#     adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    adam = tf.keras.optimizers.Adam()
    model_cnn.compile(loss=nmae_10, optimizer=adam)
    early_stop = EarlyStopping(monitor='loss', patience=5)
    history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])
#     history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)


    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.legend(['loss','val_loss'])
    plt.show()
    
    #확인용 데이터 생성
    pred_y = model_cnn.predict(x_submit)
    df_data_site['pred'] = pred_y.astype(int).tolist()
    df_data_site['new_pred'] = df_data_site.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_submit[site] = df_data_site['new_pred'].values
    df_answer[site] = (y_submit).astype(int).tolist()
    
    pred_y = model_cnn.predict(x_valid)
    df_test['pred'] = pred_y.astype(int).tolist()
    df_test['new_pred'] = df_test.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_submit_valid[site] = df_test['new_pred'].values
    df_answer_valid[site] = (y_valid).astype(int).tolist()
    
    #제출 데이터 생성
    pred_real = model_cnn.predict(x_test)
    df_real['pred'] = pred_real.astype(int).tolist() 
    df_real['new_pred'] = df_real.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_real_submit[site] = df_real['new_pred'].values
    
print('Done','#'*60)
print('Total',':',round(sola_nmae(df_answer,df_submit),2),'/','Valid',':',round(sola_nmae(df_answer_valid,df_submit_valid),2) )


# -

#제출 코드 
df_sample = pd.read_csv('data/sample_submission.csv')
df_sample['dangjin_warehouse'][:672] = df_real_submit['dangjin_warehouse'].values.tolist()
df_sample['dangjin_floating'][:672] = df_real_submit['dangjin_floating'].values.tolist()
df_sample['dangjin'][:672] = df_real_submit['dangjin'].values.tolist()
df_sample['ulsan'][:672] = df_real_submit['ulsan'].values.tolist()
print(df_sample.head(24))
print(df_sample.info())
print(len(df_sample))
df_sample.to_csv('df_result.csv',index = False)

# +
print(df_submit[:24])
print(df_answer[:24])

plt.plot(df_submit['dangjin_floating'][:24*10])
plt.plot(df_answer['dangjin_floating'][:24*10])
plt.legend(['pred','real'])
plt.show()

plt.plot(df_submit['dangjin'][:24*10])
plt.plot(df_answer['dangjin'][:24*10])
plt.legend(['pred','real'])
plt.show()

plt.plot(df_submit['dangjin_warehouse'][:24*10])
plt.plot(df_answer['dangjin_warehouse'][:24*10])
plt.legend(['pred','real'])
plt.show()

plt.plot(df_submit['ulsan'][:24*10])
plt.plot(df_answer['ulsan'][:24*10])
plt.legend(['pred','real'])
plt.show()
# -

# # 발전량 높은 구간 낮은 구간 따로 예측

# +
df_data = pd.read_csv('data/df_data.csv')
df_real_test = pd.read_csv('data/df_real_test.csv')
lst_site = df_data['variable'].unique()
df_data_sum = df_data[['variable','location','fcst_date','value']].groupby(['variable','location','fcst_date'], as_index=False).sum()
df_data_sum.columns = ['variable','location','fcst_date','val_day']
df_data = pd.merge(df_data,df_data_sum)


# lst_site = ['dangjin']
df_submit = pd.DataFrame()
df_answer = pd.DataFrame()
df_submit_valid = pd.DataFrame()
df_answer_valid = pd.DataFrame()
df_real_submit = pd.DataFrame()
# df_submit['time'] = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-31 23:00:00', freq='H')
# df_answer['time'] = pd.date_range(start='2021-01-01 00:00:00', end='2021-01-31 23:00:00', freq='H')
df_real_submit['time'] = pd.date_range(start='2021-02-01 00:00:00', end='2021-02-28 23:00:00', freq='H') #제출용
lst_all_features = ['fcst_date','fcst_hour','altitude','radiation','Temperature','Humidity','WindSpeed','WindDirection','Cloud','rain','rain_6h','rain_prob','snow_6h']
lst_features = ['fcst_hour','altitude','WindSpeed','Temperature','Humidity','Cloud','rain','rain_prob','rain_6h','snow_6h']

mx_alt = 90
mx_angle = 365
mx_rad = 1000
mx_temp = 35
mx_hum = 100
mx_windsp = 34
mx_cloud =4
mx_rain = 2
mx_rain_6 =85
mx_snow_6 =2.5
mx_rain_prob = 90
CP = {
    'dangjin_floating':1000, # 당진수상태양광 발전용량
    'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
    'dangjin':1000, # 당진태양광 발전용량
    'ulsan':500 # 울산태양광 발전용량
}
for site in lst_site:

    capacity =CP[site]
    print(site,'/',capacity,'#'*60)
    df_data_site = df_data[df_data['variable']==site].copy()
    df_data_site = df_data_site[ (df_data_site['fcst_date']<'2021-02-01') & (df_data_site['fcst_date']!='2018-03-17') & (df_data_site['fcst_date']!='2018-03-18') & (df_data_site['fcst_date']!='2018-03-19') & (df_data_site['fcst_date']!='2020-06-26')&(df_data_site['fcst_date']!='2020-06-27')  ]
 
#     mx_val = df_data_site['value'].max()
#     print('max_val :',mx_val)
    
    df_data_site= df_data_site[lst_all_features+['value']]
    df_data_site = df_data_site.sort_values(['fcst_date','fcst_hour'], axis = 0)    
    df_data_site['altitude'] = df_data_site['altitude']/mx_alt
#     df_data_site['angle'] = df_data_site['angle']/mx_angle
    df_data_site['radiation'] = df_data_site['radiation']/mx_rad
    df_data_site['Temperature'] = df_data_site['Temperature']/mx_temp
    df_data_site['Humidity'] = df_data_site['Humidity']/mx_hum
    df_data_site['WindSpeed'] = df_data_site['WindSpeed']/mx_windsp
    df_data_site['Cloud'] = df_data_site['Cloud']/mx_cloud
    df_data_site['fcst_hour'] = df_data_site['fcst_hour']/24
    df_data_site['rain'] = df_data_site['rain']/mx_rain
    df_data_site['rain_6h'] = df_data_site['rain_6h']/mx_rain_6
    df_data_site['rain_prob'] = df_data_site['rain_prob']/mx_rain_prob
    df_data_site['snow_6h'] = df_data_site['snow_6h']/mx_snow_6
#     df_data_site['value'] = df_data_site['value']/mx_val

    
    df_train = df_data_site[df_data_site['fcst_date']<'2021-01-01'].copy()
    print(len(df_train))
#     df_train = pd.concat([df_train[df_train['value']>0],df_train[df_train['value']==0].sample(frac=0.4,random_state=1004)],axis = 0)
    df_train = df_train[df_train['value']>capacity*0.1]
    print(len(df_train))
    df_test = df_data_site[(df_data_site['fcst_date']>='2021-01-01')&(df_data_site['fcst_date']<'2021-02-01')].copy()
    df_test = df_test.sort_values(['fcst_date','fcst_hour'], axis = 0)
    
    #제출용 데이터
    df_real = df_real_test[df_real_test['variable']==site].copy()
    df_real['altitude'] = df_real['altitude']/mx_alt
#     df_real['angle'] = df_real['angle']/mx_angle
    df_real['radiation'] = df_real['radiation']/mx_rad
    df_real['Temperature'] = df_real['Temperature']/mx_temp
    df_real['Humidity'] = df_real['Humidity']/mx_hum
    df_real['WindSpeed'] = df_real['WindSpeed']/mx_windsp
    df_real['Cloud'] = df_real['Cloud']/mx_cloud
    df_real['fcst_hour'] = df_real['fcst_hour']/24
    df_real['rain'] = df_real['rain']/mx_rain
    df_real['rain_6h'] = df_real['rain_6h']/mx_rain_6
    df_real['rain_prob'] = df_real['rain_prob']/mx_rain_prob
    df_real['snow_6h'] = df_real['snow_6h']/mx_snow_6

    df_real= df_real[lst_all_features]
    df_real = df_real.sort_values(['fcst_date','fcst_hour'], axis = 0)    
    
    
    x_train = np.array(df_train[lst_features].values)
    x_valid = np.array(df_test[lst_features].values)
    x_submit = np.array(df_data_site[lst_features].values)
    x_test = np.array(df_real[lst_features].values)
#     y_train = np.array(df_train['value'].values).astype('int64')
#     y_valid = np.array(df_test['value'].values).astype('int64')
#     y_submit = np.array(df_data_site['value'].values.astype('int64'))
    y_train = np.array(df_train['value'].values).astype('float64')
    y_valid = np.array(df_test['value'].values).astype('float64')
    y_submit = np.array(df_data_site['value'].values.astype('float64'))
    print('Train :', x_train.shape, '/',y_train.shape)
    print('Vaild :', x_valid.shape, '/',y_valid.shape)
    
#     print(df_train[lst_features+['value']].head(24))
    
    #모델 생성
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    import tensorflow.keras.backend as K 
    from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D, Dropout, Flatten, LSTM
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt

    EPOCHS = 1000
    BATCH_SIZE = 164 #24*3

    dt_rows = x_train.shape[1] #feature종류

#     model_cnn = Sequential()
#     model_cnn.add(Dense(128, input_dim =dt_rows))  
#     model_cnn.add(Dense(256, activation='relu'))  
# #     model_cnn.add(Dense(2048,activation='relu'))
# #     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1))
#     model_cnn.summary()


    model_cnn = Sequential()
    model_cnn.add(Dense(256, input_dim =dt_rows))   
    model_cnn.add(Dense(512,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(1024,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(256,activation=tf.keras.layers.ReLU()))
    model_cnn.add(Dense(1,activation=tf.keras.layers.ReLU()))
    model_cnn.summary()

#     model_cnn = Sequential()
#     model_cnn.add(Dense(256, input_dim =dt_rows,activation = 'relu'))  
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dense(256,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(256,activation='relu'))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(1024,activation='relu'))
#     model_cnn.add(Dense(256,activation='relu'))
#     model_cnn.add(Dense(1,activation='relu'))
#     model_cnn.summary()
    
    
#     adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    adam = tf.keras.optimizers.Adam()
    model_cnn.compile(loss='mae', optimizer=adam)
    early_stop = EarlyStopping(monitor='loss', patience=5)
    history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])
#     history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)


    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.legend(['loss','val_loss'])
    plt.show()
    
    #확인용 데이터 생성
    pred_y = model_cnn.predict(x_submit)
    df_data_site['pred'] = pred_y.astype(int).tolist()
    df_data_site['new_pred'] = df_data_site.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_submit[site] = df_data_site['new_pred'].values
    df_answer[site] = (y_submit).astype(int).tolist()
    
    pred_y = model_cnn.predict(x_valid)
    df_test['pred'] = pred_y.astype(int).tolist()
    df_test['new_pred'] = df_test.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_submit_valid[site] = df_test['new_pred'].values
    df_answer_valid[site] = (y_valid).astype(int).tolist()
    
    #제출 데이터 생성
    pred_real = model_cnn.predict(x_test)
    df_real['pred'] = pred_real.astype(int).tolist() 
    df_real['new_pred'] = df_real.apply(lambda x : 0 if x['altitude']==0 else x['pred'][0], axis = 1)
    df_real_submit[site] = df_real['new_pred'].values
    ㅁ
print('Done','#'*60)
print('Total',':',round(sola_nmae(df_answer,df_submit),2),'/','Valid',':',round(sola_nmae(df_answer_valid,df_submit_valid),2) )


# -

# # 24시간 한번에 예측 (24구간 예측)

# +
#사이트별로 개별 모델 생성
df_data = pd.read_csv('data/df_data.csv')
# lst_site = df_djfloat['variable'].unique()
# lst_site = ['dangjin']
lst_site = ['ulsan']
mx_alt = 90
mx_angle = 365
mx_rad = 1000
mx_temp = 35
mx_hum = 100
mx_windsp = 34
mx_cloud =4

for site in lst_site:
    df_data_site = df_data[df_data['variable']==site].copy()
    # df_djfloat = df_data[df_data['variable']=='dangjin_floating'].copy()
    # df_djware = df_data[df_data['variable']=='dangjin_warehouse'].copy()
    # df_dj = df_data[df_data['variable']=='dangjin'].copy()
    # df_us = df_data[df_data['variable']=='ulsan'].copy()
#     print(df_data_site['value'].max())
#     scale_value = df_data_site['value'].max()

    df_data_site= df_data_site[['fcst_date','fcst_hour','altitude','angle','radiation','Temperature','Humidity','WindSpeed','Cloud','value']]
    df_data_site = df_data_site.sort_values(['fcst_date','fcst_hour'], axis = 0)
#     df_data_site['value'] = df_data_site['value']
    df_data_site['altitude'] = df_data_site['altitude']/mx_alt
#     df_data_site['angle'] = df_data_site['angle']/mx_angle
    df_data_site['radiation'] = df_data_site['radiation']/mx_rad
    df_data_site['Temperature'] = df_data_site['Temperature']/mx_temp
    df_data_site['Humidity'] = df_data_site['Humidity']/mx_hum
    df_data_site['WindSpeed'] = df_data_site['WindSpeed']/mx_windsp
    df_data_site['Cloud'] = df_data_site['Cloud']/mx_cloud


    
#     print(df_data_site.info())
#     print('#'*70)
    lst_date = df_data_site['fcst_date'].unique()

    lst_x = []
    lst_y = []
    for date in lst_date:
        df_tmp = df_data_site[df_data_site['fcst_date']==date].copy()
        if len(df_tmp) == 24:
    #         lst_x.append(df_tmp[['altitude','angle','radiation','Temperature','Humidity','WindSpeed','Cloud']].transpose().values)
            lst_x.append(df_tmp[['altitude','radiation','Humidity','WindSpeed','Cloud']].transpose().values)
            lst_y.append(df_tmp['value'].transpose().values)


    x_data = np.array(lst_x)
    y_data = np.array(lst_y)
    

    print('X Data : ',x_data.shape)
    print('Y Data : ',y_data.shape)

    train_split = int(x_data.shape[0]*0.8)
    x_train = x_data[:train_split]
    x_valid = x_data[train_split:]
    y_train = y_data[:train_split]
    y_valid = y_data[train_split:]
#     y_train = y_data[:train_split].astype('int64')
#     y_valid = y_data[train_split:].astype('int64')
    print('Train :', x_train.shape, '/',y_train.shape)
    print('Train :', x_valid.shape, '/',y_valid.shape)
    
    
    
    
    
    #모델 생성
    import tensorflow as tf
    from tensorflow.keras.models import Sequential 
    import tensorflow.keras.backend as K 
    from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D, Dropout, Flatten
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt

    EPOCHS = 200
    BATCH_SIZE = 72

    dt_rows = x_train.shape[1] #feature종류
    dt_cols = x_train.shape[2] #timestep

#     model_cnn = Sequential()
#     model_cnn.add(Conv1D(24,  kernel_size=1, padding = 'causal', input_shape =(dt_rows, dt_cols),activation='sigmoid'))
#     model_cnn.add(Conv1D(32,  kernel_size=3, padding = 'causal', activation='relu'))
#     model_cnn.add(Flatten())
#     model_cnn.add(Dense(512,activation='relu'))
#     model_cnn.add(Dense(120,activation='relu'))
#     model_cnn.add(Dropout(0.2))
#     model_cnn.add(Dense(24, activation = 'softmax'))
#     model_cnn.summary()

    model_cnn = Sequential()
    model_cnn.add(Conv1D(24,  kernel_size=2, padding = 'causal', input_shape =(dt_rows, dt_cols),activation='relu'))
    model_cnn.add(Conv1D(48,  kernel_size=2, padding = 'causal', activation='relu'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(1024,activation='relu'))
    model_cnn.add(Dense(512,activation='relu'))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Dense(24))
    model_cnn.summary()
    
    


    model_cnn.compile(loss=tf.keras.losses.MAE, optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    # tf.keras.optimizers.Adam(learning_rate=1e-5)
    history_cnn = model_cnn.fit(x_train,y_train, validation_data =(x_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)

    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.legend(['loss','val_loss'])
    plt.show()




