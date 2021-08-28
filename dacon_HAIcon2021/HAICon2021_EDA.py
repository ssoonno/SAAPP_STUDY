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

# # HAICon2021 EDA 

# ## 필요 Library load

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import pytz
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import seaborn as sns
from IPython.core.display import display,HTML 
display(HTML("<style>.container { width:80% !important; }</style>"))


# ### Data load 및 data 확인
# 컬럼은 CO~C79 , Target은 attack
# 날짜별로 train/valid/test가 존재하는것이 아닌 같은 일자에도 Train/Test가 섞여있다. 데이터셋분리는 일자별이 아니다.

# +
# df_train = pd.read_csv('data/train/train1.csv')

#데이터 일자 확인
print('Check data period')
print('train1.csv: ',pd.read_csv('data/train/train1.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train1.csv')['timestamp'].max())
print('train2.csv: ',pd.read_csv('data/train/train2.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train2.csv')['timestamp'].max())
print('train3.csv: ',pd.read_csv('data/train/train3.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train3.csv')['timestamp'].max())
print('train4.csv: ',pd.read_csv('data/train/train4.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train4.csv')['timestamp'].max())
print('train5.csv: ',pd.read_csv('data/train/train5.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train5.csv')['timestamp'].max())
print('train6.csv: ',pd.read_csv('data/train/train6.csv')['timestamp'].min(),'~',pd.read_csv('data/train/train6.csv')['timestamp'].max())

print('validation.csv: ',pd.read_csv('data/validation/validation.csv')['timestamp'].min(),'~',pd.read_csv('data/validation/validation.csv')['timestamp'].max())
print('test1.csv: ',pd.read_csv('data/test/test1.csv')['timestamp'].min(),'~',pd.read_csv('data/test/test1.csv')['timestamp'].max())
print('test2.csv: ',pd.read_csv('data/test/test2.csv')['timestamp'].min(),'~',pd.read_csv('data/test/test2.csv')['timestamp'].max())
print('test3.csv: ',pd.read_csv('data/test/test3.csv')['timestamp'].min(),'~',pd.read_csv('data/test/test3.csv')['timestamp'].max())


#데이터 형태 파악
df_train = pd.concat([pd.read_csv('data/train/train1.csv'),
                      pd.read_csv('data/train/train2.csv'),
                      pd.read_csv('data/train/train3.csv'),
                      pd.read_csv('data/train/train4.csv'),
                      pd.read_csv('data/train/train5.csv'),
                      pd.read_csv('data/train/train6.csv')])
df_valid = pd.read_csv('data/validation/validation.csv')
df_test = pd.concat([pd.read_csv('data/test/test1.csv'),pd.read_csv('data/test/test2.csv'),pd.read_csv('data/test/test3.csv')])
print(df_valid.head(3))

#데이터 현황 확인
df_datecnt_tr = pd.DataFrame(df_train['timestamp'])
df_datecnt_tr['date']= df_datecnt_tr['timestamp'].str.slice(0,10)
df_datecnt_vld = pd.DataFrame(df_valid['timestamp'])
df_datecnt_vld['date']= df_datecnt_vld['timestamp'].str.slice(0,10)
df_datecnt_ts = pd.DataFrame(df_test['timestamp'])
df_datecnt_ts['date']= df_datecnt_ts['timestamp'].str.slice(0,10)

df_datecnt_tr = df_datecnt_tr.groupby('date').count().reset_index()
df_datecnt_tr.columns = ['date','train']
df_datecnt_vld = df_datecnt_vld.groupby('date').count().reset_index()
df_datecnt_vld.columns = ['date','valid']
df_datecnt_ts = df_datecnt_ts.groupby('date').count().reset_index()
df_datecnt_ts.columns = ['date','test']

df_plt_cnt = pd.merge(df_datecnt_tr,df_datecnt_vld, on =['date'],how='outer')
df_plt_cnt = pd.merge(df_plt_cnt,df_datecnt_ts, on =['date'],how='outer')
df_plt_cnt = df_plt_cnt.fillna(0)
df_plt_cnt.set_index('date',inplace=True)
df_plt_cnt = df_plt_cnt.sort_index()
df_plt_cnt.plot.bar(stacked=True,title='Data distribution by date')

del([df_plt_cnt,df_datecnt_tr,df_datecnt_vld,df_datecnt_ts])
gc.collect()
# -

# ### Attack 여부에 따른 각 변수의 데이터 분포의 차이가 있는지 확인
# Target의 그룹별 데이터의 imbalance가 심함(99.993%:0.0007%)<br>
# Attack=1 시의 데이터가 적어 그룹별 데이터의 분포 차이로 변수별 중요도를 확인할수는 없으나 데이터의 전반적인 분포 성향을 보기위해 Attack의 그룹별 데이터 분포 확인<br>
# (과거버젼)육안으로 확인시 C01, C05, C24, C39, C44, C79의 변수의 그룹별 차이가 확인됨<br>
# (현재버젼)

# +
import warnings
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


df_train['attack'] = 0
df_tot =  pd.concat([df_train,df_valid])
print('Row Train/Test:',len(df_train),'/',len(df_valid))
print('Row mergerd data:',len(df_tot))
print('Group size ',df_tot.groupby('attack').size())


warnings.filterwarnings(action='ignore')
lst_cols = df_tot.columns.difference(['timestamp','attack'])
ax_r =0 
ax_c =0
fig,axes = plt.subplots(18,5,figsize = (200,150))
sns.set(font_scale=2) # 아주 크게 
df_anova = pd.DataFrame(columns = ['column','P-value','F-stat'])

for col in lst_cols:
    try:
        print(col)
        anova_result = anova_lm(ols((col+'~C(attack)'),df_tot).fit())
        df_anova = df_anova.append({'column': col, 'P-value':round(anova_result['PR(>F)'][0],3),'F-stat':round(anova_result['F'][0],3)}, ignore_index =True)
        sns.distplot(df_tot[df_tot['attack']==0][col],color ='blue',label='Normal', ax= axes[ax_r,ax_c])
        sns.distplot(df_tot[df_tot['attack']==1][col],color='red',label='Attack' , ax= axes[ax_r,ax_c]).set_xlabel(col+'-'+str(round(anova_result['PR(>F)']['C(attack)'],3)),fontsize=70)

        if ax_c <4:
            ax_c+=1
        else :
            ax_r+=1
            ax_c=0
    except Exception as e:
        print('Exception column ',col)
        print(e)
        if ax_c <4:
            ax_c+=1
        else :
            ax_r+=1
            ax_c=0

plt.legend(title="Attacked")
plt.show()
        

# +
import warnings
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import MinMaxScaler

lst_features = df_train.columns.difference(['timestamp','attack'])
scaler = MinMaxScaler()
scaler.fit(df_train[lst_features])
df_train['attack'] = 0
df_m_tr = pd.DataFrame(scaler.transform(df_train[lst_features]), columns = lst_features,index =df_train.index) #feature scaling
df_m_tr = pd.concat([df_train[['timestamp','attack']],df_m_tr],axis = 1) #concat time tag
df_m_vld = pd.DataFrame(scaler.transform(df_valid[lst_features]), columns=lst_features, index = df_valid.index) #feature scaling
df_m_vld = pd.concat([df_valid[['timestamp','attack']],df_m_vld],axis =1) #concat time tag,attack
df_tot = pd.concat([df_m_tr,df_m_vld])
del([df_m_tr,df_m_vld])
gc.collect()

print('Row Train/Test:',len(df_train),'/',len(df_valid))
print('Row mergerd data:',len(df_tot))
print('Group size ',df_tot.groupby('attack').size())


warnings.filterwarnings(action='ignore')
lst_cols = df_tot.columns.difference(['timestamp','attack'])
ax_r =0 
ax_c =0
fig,axes = plt.subplots(18,5,figsize = (200,150))
sns.set(font_scale=2) # 아주 크게 
df_anova = pd.DataFrame(columns = ['column','P-value','F-stat'])

for col in lst_cols:
    try:
        print(col)
        anova_result = anova_lm(ols((col+'~C(attack)'),df_tot).fit())
        df_anova = df_anova.append({'column': col, 'P-value':round(anova_result['PR(>F)'][0],3),'F-stat':round(anova_result['F'][0],3)}, ignore_index =True)
        sns.distplot(df_tot[df_tot['attack']==0][col],color ='blue',label='Normal', ax= axes[ax_r,ax_c])
        sns.distplot(df_tot[df_tot['attack']==1][col],color='red',label='Attack' , ax= axes[ax_r,ax_c]).set_xlabel(col+'-'+str(round(anova_result['PR(>F)']['C(attack)'],3)),fontsize=70)

        if ax_c <4:
            ax_c+=1
        else :
            ax_r+=1
            ax_c=0
    except Exception as e:
        print('Exception column ',col)
        print(e)
        if ax_c <4:
            ax_c+=1
        else :
            ax_r+=1
            ax_c=0

plt.legend(title="Attacked")
plt.show()
        
# -

print('Total/Filtered:',len(df_anova),'/',len(df_anova[df_anova['P-value']<0.05]))
df_anova[df_anova['P-value']<0.05]
# df_tot.head(10)

# ### 변수별 트렌드 및 Attack여부에 따른 트렌드 변화 확인
# Attack 시점에 이상현상을 보이는 변수들 C01,C02,C04,C07,C10(일부),C13,C15,C26,C27,C32,C44(일부),C45,C46(일부),C47,C49,C55,C66,C70,C72,C73,C75<br>
# 각변수들의 변화량 데이터 확인시 이상현상을 보이는 변수들만 Attack시점에 이상을 보임<br>
# (신규) C16, C24, C31, C32, C43(일부), C59, C68, C70, C71(일부), C73, C75, C76, C77, C84

# +
df_train['attack'] = 0
df_valid['timestamp'] = pd.to_datetime(df_valid['timestamp'])
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
df_tot =  pd.concat([df_train,df_valid])

lst_cols = df_tot.columns.difference(['timestamp','attack'])
for col in lst_cols:
    sns.set(style="darkgrid")
    sns.relplot(x='timestamp', y=col ,hue = 'attack', data=df_valid, height=7,aspect =4/1)
 


# +
#변화량 트렌드 확인
df_trend = df_valid.copy()
df_trend =df_trend.sort_values('timestamp')
lst_cols = df_trend.columns.difference(['timestamp','attack'])

df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'])

for col in lst_cols:
    df_trend[col+'_diff']= df_trend[col] -df_trend[col].shift(1)
df_trend['diff_time'] = pd.to_numeric((df_trend['timestamp']-df_trend['timestamp'].shift(1)).astype('timedelta64[s]'))


df_trend = df_trend.dropna()
# print(df_trend.head(10))
print('Time diff is not 1second : ', len(df_trend[df_trend['diff_time']!=1])) 

lst_cols = df_trend.columns.difference(['timestamp','attack','diff_time'])
lst_cols = [value for value in lst_cols if '_diff' in value]
df_trend = df_trend[['timestamp','attack']+lst_cols]
for col in lst_cols:
    sns.set(style="darkgrid")
    sns.relplot(x='timestamp', y=col ,hue = 'attack', data=df_trend, height=7,aspect =4/1)
    
del(df_trend)
gc.collect()
#C01,C02,C07,C10(일부),C13,C15,C26,C27,C32,C44(일부),C45,C46(일부),C47,C49,C55,C66,C70,C72,C73,C75
# +

lst_features = df_train.columns.difference(['timestamp','attack'])
scaler = MinMaxScaler()
scaler.fit(df_train[lst_features])

df_m_vld = pd.DataFrame(scaler.transform(df_valid[lst_features]), columns=lst_features, index = df_valid.index) #feature scaling
df_m_vld = pd.concat([df_valid[['timestamp','attack']],df_m_vld],axis =1) #concat time tag,attack


lst_cols = df_tot.columns.difference(['timestamp','attack'])
for col in lst_cols:
    sns.set(style="darkgrid")
    sns.relplot(x='index', y=col ,hue = 'attack', data=df_m_vld.reset_index(), height=7,aspect =4/1)
 
# -





