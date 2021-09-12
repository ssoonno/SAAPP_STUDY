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

# # HAICon2021_Modeling

# ## 필요 라이브러리 Import

# +
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import pytz
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import seaborn as sns
import cv2
from IPython.core.display import display,HTML 
from TaPR_pkg import etapr
from sklearn.preprocessing import MinMaxScaler

# TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)#평가 방법
display(HTML("<style>.container { width:80% !important; }</style>"))

# +
#필요 데이터 로드
df_train = pd.concat([pd.read_csv('data/train/train1.csv'),
                      pd.read_csv('data/train/train2.csv'),
                      pd.read_csv('data/train/train3.csv'),
                      pd.read_csv('data/train/train4.csv'),
                      pd.read_csv('data/train/train5.csv'),
                      pd.read_csv('data/train/train6.csv')])
df_train = df_train.reset_index()
df_valid = pd.read_csv('data/validation/validation.csv')
df_valid = df_valid.reset_index()
#PC성능 문제로 Test Data분할
# df_test = pd.concat([pd.read_csv('data/test/test1.csv'),pd.read_csv('data/test/test2.csv'),pd.read_csv('data/test/test3.csv'),pd.read_csv('data/test/test4.csv')])
df_test1 = pd.read_csv('data/test/test1.csv')
df_test2 = pd.read_csv('data/test/test2.csv')
df_test3 = pd.read_csv('data/test/test3.csv')

print('Train rows: ',len(df_train))
print('Valid rows: ',len(df_valid))
print('Test1 rows: ',len(df_test1))
print('Test2 rows: ',len(df_test2))
print('Test3 rows: ',len(df_test3))

# -


# # 1.Simple AutoEncoder

# ### Preprocess

# +
#중요 변수 C01,C02,C07,C10(일부),C13,C15,C26,C27,C32,C44(일부),C45,C46(일부),C47,C49,C55,C66,C70,C72,C73,C75
str_target = 'attack'
# lst_features = ['C01','C02','C07','C10','C13','C15','C26','C27','C32','C44','C45','C46','C47','C49','C55','C66','C70','C72','C73','C75']
# lst_features = [ 'C16', 'C24', 'C31', 'C32', 'C43', 'C59', 'C68', 'C70', 'C71', 'C73', 'C75', 'C76', 'C77', 'C84']
# lst_features = df_train.columns.difference(['index','timestamp','attack'])
# lst_features = set(lst_features).difference(set(['C08', 'C09', 'C10', 'C17', 'C18', 'C19', 'C22', 'C26', 'C29', 'C34', 'C36', 'C38', 'C39', 'C46', 'C48', 'C49', 'C52', 'C55', 'C61', 'C63', 'C64', 'C79', 'C82', 'C85']))
lst_features = ['C01', 'C02', 'C07', 'C30', 'C41', 'C44', 'C62', 'C03', 'C12', 'C16', 'C24', 'C31', 'C32', 'C43', 'C60', 'C68', 'C70', 'C71', 'C73', 'C75', 'C76', 'C77', 'C84']
print('Feature list length:',len(lst_features))

scaler = MinMaxScaler()
scaler.fit(df_train[lst_features])
df_m_tr = pd.DataFrame(scaler.transform(df_train[lst_features]), columns = lst_features,index =df_train.index) #feature scaling
df_m_tr = pd.concat([df_train[['timestamp']],df_m_tr],axis = 1) #concat time tag
df_m_vld = pd.DataFrame(scaler.transform(df_valid[lst_features]), columns=lst_features, index = df_valid.index) #feature scaling
df_m_vld = pd.concat([df_valid[['timestamp','attack']],df_m_vld],axis =1) #concat time tag,attack


# df_m_ts = pd.DataFrame(scaler.transform(df_test[lst_features]), columns = lst_features, index=df_test.index) #feature scaling
# df_m_ts = pd.concat([df_test[['time']],df_m_ts],axis = 1) #concat time tag
df_m_ts1 = pd.DataFrame(scaler.transform(df_test1[lst_features]), columns = lst_features, index=df_test1.index) #feature scaling
df_m_ts1 = pd.concat([df_test1[['timestamp']],df_m_ts1],axis = 1) #concat time tag

df_m_ts2 = pd.DataFrame(scaler.transform(df_test2[lst_features]), columns = lst_features, index=df_test2.index) #feature scaling
df_m_ts2 = pd.concat([df_test2[['timestamp']],df_m_ts2],axis = 1) #concat time tag

df_m_ts3 = pd.DataFrame(scaler.transform(df_test3[lst_features]), columns = lst_features, index=df_test3.index) #feature scaling
df_m_ts3 = pd.concat([df_test3[['timestamp']],df_m_ts3],axis = 1) #concat time tag

# # Create diff value
# for col in lst_features:
#     df_m_tr[col+'_diff']= df_m_tr[col] -df_m_tr[col].shift(1)
#     df_m_vld[col+'_diff']= df_m_vld[col] -df_m_vld[col].shift(1)
#     df_m_ts1[col+'_diff']= df_m_ts1[col] -df_m_ts1[col].shift(1)
#     df_m_ts2[col+'_diff']= df_m_ts2[col] -df_m_ts2[col].shift(1)
#     df_m_ts3[col+'_diff']= df_m_ts3[col] -df_m_ts3[col].shift(1)

df_m_tr = df_m_tr.fillna(0)
df_m_vld = df_m_vld.fillna(0)
df_m_ts1 = df_m_ts1.fillna(0)
df_m_ts2 = df_m_ts2.fillna(0)
df_m_ts3 = df_m_ts3.fillna(0)

lst_features = df_m_tr.columns.difference(['index','timestamp','attack'])


np_tr = df_m_tr[lst_features].to_numpy()
np_tr_1 = np_tr[:int(np_tr.shape[0]*0.8)]
np_tr_2 = np_tr[int(np_tr.shape[0]*0.8):]
np_vld = df_m_vld[lst_features].to_numpy()
np_ts1 = df_m_ts1[lst_features].to_numpy()
np_ts2 = df_m_ts2[lst_features].to_numpy()
np_ts3 = df_m_ts3[lst_features].to_numpy()

del([df_train,df_valid,df_test1,df_test2,df_test3])
print(np_tr.shape)


# -

# ### Model Create & Train

# +
import tensorflow as tf
from tensorflow.keras.models import Sequential 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='cpu')

EPOCHS = 100
BATCH_SIZE = 60*30 #24*3

#V01
# simple_AE = Sequential()
# simple_AE.add(Dense(1024,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(256,activation = 'relu'))
# simple_AE.add(Dense(1,activation = 'relu'))
# simple_AE.add(Dense(256,activation = 'relu'))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()


#V02
# simple_AE = Sequential()
# simple_AE.add(Dense(14,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(8,activation = 'relu'))
# simple_AE.add(Dense(4,activation = 'relu'))
# simple_AE.add(Dense(1,activation = 'relu'))
# simple_AE.add(Dense(4,activation = 'relu'))
# simple_AE.add(Dense(8,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()

#V03
# simple_AE = Sequential()
# simple_AE.add(Dense(256,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(256,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()


#V04
# simple_AE = Sequential()
# simple_AE.add(Dense(256,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(2048,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(256,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()

#V05(+diff_tag)
# simple_AE = Sequential()
# simple_AE.add(Dense(256,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(256,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()



# #V06(all tag)
# simple_AE = Sequential()
# simple_AE.add(Dense(512,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(2048,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()

# #V07
# simple_AE = Sequential()
# simple_AE.add(Dense(512,input_dim=(np_tr.shape[1])))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(2048,activation = 'relu'))
# simple_AE.add(Dense(1024,activation = 'relu'))
# simple_AE.add(Dense(512,activation = 'relu'))
# simple_AE.add(Dense(np_tr.shape[1]))
# simple_AE.summary()

#V08
simple_AE = Sequential()
simple_AE.add(Dense(512,input_dim=(np_tr.shape[1])))
simple_AE.add(Dense(1024,activation = 'relu'))
simple_AE.add(Dense(2048,activation = 'relu'))
simple_AE.add(Dense(4096,activation = 'relu'))
simple_AE.add(Dense(2048,activation = 'relu'))
simple_AE.add(Dense(1024,activation = 'relu'))
simple_AE.add(Dense(512,activation = 'relu'))
simple_AE.add(Dense(np_tr.shape[1]))
simple_AE.summary()



adam = tf.keras.optimizers.Adam()
simple_AE.compile(loss='mae',optimizer=adam)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_simple_AE = simple_AE.fit(np_tr_1,np_tr_1,validation_data=(np_tr_2,np_tr_2), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])

plt.plot(history_simple_AE.history['loss'])
plt.plot(history_simple_AE.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()

simple_AE.save('model/Simple_AE.h5') #model save
# -

# ### Plot anomally result with real attack tag

# +
import seaborn as sns
from tensorflow.keras.models import load_model

simple_AE = load_model('model/Simple_AE.h5')
np_vld_pred = simple_AE.predict(np_vld)
print('Total MAE:',np.mean(np.abs(np_vld - np_vld_pred)))
print('Feature MAE:',np.mean(np.abs(np_vld - np_vld_pred),axis =0))


df_vld_result = pd.concat([pd.DataFrame(data = np.abs(np_vld - np_vld_pred),columns = lst_features),
                          pd.DataFrame(data = np.mean(np.abs(np_vld - np_vld_pred),axis =1), columns = ['TOT_MAE'])],axis =1)
df_vld_result =pd.concat([df_m_vld[['timestamp','attack']],df_vld_result],axis =1)
df_vld_result = df_vld_result.sort_values('timestamp')
sns.relplot(x='index', y = 'TOT_MAE', hue = 'attack', data =df_vld_result.reset_index(),height=7, aspect = 4/1)
# sns.scatterplot(x='index',y='TOT_MAE', hue='attack', data = df_vld_result.reset_index(), height=7)
# del([df_m_tr,df_m_vld])
# del(np_vld,np_vld_pred)
# -

# ### Find best threshold with valid data

# +

thres_min = round(df_vld_result[df_vld_result['attack']==1]['TOT_MAE'].min(),3)
thres_max = round(df_vld_result[df_vld_result['attack']==1]['TOT_MAE'].max(),3)
df_model_result = pd.DataFrame(columns = ['Threshold','Precision','Recall','F1','TaPR'])


print('Threshold range :',round(thres_min,3),'~',round(thres_max,3))

for threshold in np.arange(thres_min,thres_max,0.001):
    try:
        df_vld_result['PRED'] = df_vld_result.apply(lambda x :1 if x['TOT_MAE']>threshold else 0,axis =1)
        precisiton =len(df_vld_result[(df_vld_result['attack']==1)&(df_vld_result['PRED']==1)])/len(df_vld_result[df_vld_result['PRED']==1])
        recall = len(df_vld_result[(df_vld_result['attack']==1)&(df_vld_result['PRED']==1)])/len(df_vld_result[df_vld_result['attack']==1])
        f1_score = 2*(precisiton*recall)/(precisiton+recall)
        TaPR = etapr.evaluate_haicon(anomalies=df_vld_result['attack'].values, predictions=df_vld_result['PRED'].values) #대회 오차
        print(round(threshold,3),'::'+f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        df_model_result = df_model_result.append({'Threshold':threshold,'Precision':precisiton,'Recall':recall,'F1':f1_score, 'TaPR':TaPR['f1']},ignore_index=True)
    except Exception as e:
        print("Exception Occured with threshold ", threshold)
        print(e)

print(df_model_result)

threshold = df_model_result[df_model_result['TaPR'] == df_model_result['TaPR'].max()]['Threshold'].values[0]
print('Best threshold is ',round(threshold,3))
print(df_model_result[df_model_result['Threshold']==threshold]) 
# del(df_vld_result)
# -
# ### Predict test data & make submission dataframe

# +

np_result_ts1 = simple_AE.predict(np_ts1)
df_result_ts1 = pd.concat([pd.DataFrame(data = np.abs(np_ts1 - np_result_ts1),columns = lst_features),
                          pd.DataFrame(data = np.mean(np.abs(np_ts1 - np_result_ts1),axis =1), columns = ['mae'])],axis =1)
df_result_ts1 =pd.concat([df_m_ts1['timestamp'],df_result_ts1['mae']],axis =1)

np_result_ts2 = simple_AE.predict(np_ts2)
df_result_ts2 = pd.concat([pd.DataFrame(data = np.abs(np_ts2 - np_result_ts2),columns = lst_features),
                          pd.DataFrame(data = np.mean(np.abs(np_ts2 - np_result_ts2),axis =1), columns = ['mae'])],axis =1)
df_result_ts2 =pd.concat([df_m_ts2['timestamp'],df_result_ts2['mae']],axis =1)

np_result_ts3 = simple_AE.predict(np_ts3)
df_result_ts3 = pd.concat([pd.DataFrame(data = np.abs(np_ts3 - np_result_ts3),columns = lst_features),
                          pd.DataFrame(data = np.mean(np.abs(np_ts3 - np_result_ts3),axis =1), columns = ['mae'])],axis =1)
df_result_ts3 =pd.concat([df_m_ts3['timestamp'],df_result_ts3['mae']],axis =1)

df_submission_org = pd.read_csv('data/sample_submission.csv')
df_submission = pd.concat([df_result_ts1,df_result_ts2,df_result_ts3])
df_submission = df_submission.reset_index()
df_submission['attack'] = df_submission.apply(lambda x : 1 if x['mae']>threshold else 0 ,axis =1)
sns.relplot(x = 'index', y ='mae', hue='attack',  data =df_submission.reset_index(), height=7, aspect = 4/1)
print('Examle submission : ',len(df_submission),'Created submission : ',len(df_submission_org))

# df_submission = df_submission[['timestamp','attack']]
# df_submission.to_csv('result_SimpleAE_v03_3.csv',index = False)
print("Model result save Done")
# -


df_submission = df_submission[['timestamp','attack']]
df_submission.to_csv('result_SimpleAE_v08.csv',index = False)























# # 2.Time_lag AutoEncoder

# ### Preprocess

# +
#중요 변수 C01,C02,C07,C10(일부),C13,C15,C26,C27,C32,C44(일부),C45,C46(일부),C47,C49,C55,C66,C70,C72,C73,C75
str_target = 'attack'
# lst_features = ['C01','C02','C07','C10','C13','C15','C26','C27','C32','C44','C45','C46','C47','C49','C55','C66','C70','C72','C73','C75']
lst_features = [ 'C16', 'C24', 'C31', 'C32', 'C43', 'C59', 'C68', 'C70', 'C71', 'C73', 'C75', 'C76', 'C77', 'C84']
timestep = 30

scaler = MinMaxScaler()
scaler.fit(df_train[lst_features])
df_m_tr = pd.DataFrame(scaler.transform(df_train[lst_features]), columns = lst_features,index =df_train.index) #feature scaling
df_m_tr = pd.concat([df_train[['timestamp']],df_m_tr],axis = 1) #concat time tag
df_m_vld = pd.DataFrame(scaler.transform(df_valid[lst_features]), columns=lst_features, index = df_valid.index) #feature scaling
df_m_vld = pd.concat([df_valid[['timestamp','attack']],df_m_vld],axis =1) #concat time tag,attack


# df_m_ts = pd.DataFrame(scaler.transform(df_test[lst_features]), columns = lst_features, index=df_test.index) #feature scaling
# df_m_ts = pd.concat([df_test[['time']],df_m_ts],axis = 1) #concat time tag
# df_m_ts1 = pd.DataFrame(scaler.transform(df_test1[lst_features]), columns = lst_features, index=df_test1.index) #feature scaling
# df_m_ts1 = pd.concat([df_test1[['timestamp']],df_m_ts1],axis = 1) #concat time tag

# df_m_ts2 = pd.DataFrame(scaler.transform(df_test2[lst_features]), columns = lst_features, index=df_test2.index) #feature scaling
# df_m_ts2 = pd.concat([df_test2[['timestamp']],df_m_ts2],axis = 1) #concat time tag

# df_m_ts3 = pd.DataFrame(scaler.transform(df_test3[lst_features]), columns = lst_features, index=df_test3.index) #feature scaling
# df_m_ts3 = pd.concat([df_test3[['timestamp']],df_m_ts3],axis = 1) #concat time tag



for col in lst_features:
    for lag in range(1,timestep+1):
        df_m_tr[col+'_'+str(lag)] = df_m_tr[col].shift(lag)
        df_m_vld[col+'_'+str(lag)] = df_m_vld[col].shift(lag)
#         df_m_ts1[col+'_'+str(lag)] = df_m_ts1[col].shift(lag)
#         df_m_ts2[col+'_'+str(lag)] = df_m_ts2[col].shift(lag)
#         df_m_ts3[col+'_'+str(lag)] = df_m_ts3[col].shift(lag)

df_m_tr = df_m_tr.dropna()
df_m_vld = df_m_vld.dropna()

lst_features = df_m_tr.columns.difference(['timestamp'])

np_tr = df_m_tr[df_m_tr.columns.difference(['timestamp'])].to_numpy()
np_tr_1 = np_tr[:int(np_tr.shape[0]*0.8)]
np_tr_2 = np_tr[int(np_tr.shape[0]*0.8):]
np_vld = df_m_vld[df_m_vld.columns.difference(['timestamp','attack'])].to_numpy()
# np_ts1 = df_m_ts1[df_m_ts1.columns.difference(['timestamp'])].to_numpy()
# np_ts2 = df_m_ts2[df_m_ts2.columns.difference(['timestamp'])].to_numpy()
# np_ts3 = df_m_ts3[df_m_ts3.columns.difference(['timestamp'])].to_numpy()

del([df_train,df_valid,df_test1,df_test2,df_test3])
gc.collect()
# -

# ### Make model

# +
import tensorflow as tf
from tensorflow.keras.models import Sequential 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='cpu')

EPOCHS = 100
BATCH_SIZE = 128 #24*3


#V01
TimeLag_AE = Sequential()
TimeLag_AE.add(Dense(256,input_dim=(np_tr.shape[1])))
TimeLag_AE.add(Dense(64,activation = 'relu'))
TimeLag_AE.add(Dense(1,activation = 'relu'))
TimeLag_AE.add(Dense(64,activation = 'relu'))
TimeLag_AE.add(Dense(256,activation = 'relu'))
TimeLag_AE.add(Dense(np_tr.shape[1]))
TimeLag_AE.summary()


adam = tf.keras.optimizers.Adam()
TimeLag_AE.compile(loss='mae',optimizer=adam)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_TimeLag_AE = TimeLag_AE.fit(np_tr_1,np_tr_1,validation_data=(np_tr_2,np_tr_2), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])

plt.plot(history_TimeLag_AE.history['loss'])
plt.plot(history_TimeLag_AE.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()

TimeLag_AE.save('model/TimeLag_AE.h5') #model save

# +
import seaborn as sns
from tensorflow.keras.models import load_model


np_vld_1 = np_vld[:int(np_vld.shape[0]*0.5)]
np_vld_2 = np_vld[int(np_vld.shape[0]*0.5):]

TimeLag_AE = load_model('model/TimeLag_AE.h5')
np_vld_pred1 = TimeLag_AE.predict(np_vld)
np_vld_pred2 = TimeLag_AE.predict(np_vld_2)
np_vld_pred = 
print('Total MAE:',np.mean(np.abs(np_vld - np_vld_pred)))
print('Feature MAE:',np.mean(np.abs(np_vld - np_vld_pred),axis =0))


df_vld_result = pd.concat([pd.DataFrame(data = np.abs(np_vld - np_vld_pred),columns = lst_features),
                          pd.DataFrame(data = np.mean(np.abs(np_vld - np_vld_pred),axis =1), columns = ['TOT_MAE'])],axis =1)
df_vld_result =pd.concat([df_valid[['timestamp','attack']],df_vld_result],axis =1)

sns.relplot(x='timestamp', y = 'TOT_MAE', hue = 'attack', data =df_vld_result,height=7, aspect = 4/1)

# del([df_m_tr,df_m_vld])
# del(np_vld,np_vld_pred)
# +
print(np_vld.shape)
np_vld_1 = np_vld[:int(np_vld.shape[0]*0.5)]
np_vld_2 = np_vld[int(np_vld.shape[0]*0.5):]

np_vld_1 = np_vld_1.append(np_vld_2)
np_vld_1.shape
# -



