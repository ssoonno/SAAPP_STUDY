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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import pytz
from sklearn.linear_model import LinearRegression

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


# +
#예보 데이터(강수확률, 6시간 강수량/적설, 강수형태) 추가
import unicodedata
import datetime
import os
def date_process(day, lst_ym):
    date = None
  
    if day.find('Start')>0:
        lst_ym[0] = day.split(':')[1][1:5]
        lst_ym[1] = day.split(':')[1][5:7]
    else:
    # print(lst_ym,day)
        date = datetime.date(int(lst_ym[0]),int(lst_ym[1]),int(day))
    return date

df_data = pd.DataFrame()

for file in os.listdir(u'data/예보/') :
    if file.find('.csv') >0:
#         print(file)
#         print(file.split('_')[0], file.split('_')[1], file.split('_')[2][:4], file.split('_')[2][4:])
        site = unicodedata.normalize('NFC',file.split('_')[0])
        if site =='삼산동':
            site = 'ulsan'
        else :
            site = 'dangjin'
        
        metric =  unicodedata.normalize('NFC',file.split('_')[1])
        if metric.strip() == "강수형태":
            metric = 'rain'
            
        elif metric.strip() == '강수확률':
            metric = 'rain_prob'
            
        elif metric.strip() == '6시간강수량' :
            metric = 'rain_6h'
            
        elif metric == '6시간적설':
            metric = 'snow_6h'
            
        year = file.split('_')[2][:4]
        month =  file.split('_')[2][4:]
        lst_ym = [year,month]
#         print(site, metric, year, month)
        df_file = pd.read_csv('data/예보/'+file)
        df_file.columns = ['day','hour','forecast', 'value']
        df_file['date'] = df_file.iloc[:,0].apply(lambda x : date_process(str(x),lst_ym))
        df_file =df_file.dropna()
        df_file['hour'] = (df_file['hour']/100).astype('int')
        df_file['location'] =  site
        df_file['metric'] =  metric

    df_data = pd.concat([df_data, df_file], axis=0)


df_data = df_data.pivot(index = ['location','date','hour','forecast'], columns='metric', values='value')

df_data = df_data.interpolate()
df_data = df_data.fillna(0)
print(df_data.head(10))
df_data.to_csv('data/df_fcst_manual.csv')

# +
#발전량 정보 load
df_energy = pd.read_csv('data/energy.csv')
df_energy['date'] = pd.to_datetime(df_energy['time'].str.slice(0,10)) 
df_energy['hour'] = pd.to_numeric(df_energy['time'].str.slice(11,13).str.replace(':',''))
df_energy['date'] = df_energy.apply(lambda x : fnc_makeDate(x['date'],x['hour']), axis =1)
df_energy['hour'] = df_energy.apply(lambda x : fnc_makeHour(x['hour']), axis =1)

df_energy_melted = pd.melt(df_energy,id_vars=['time','date','hour'] ) #지역별 melting
df_energy_melted = df_energy_melted[['date','hour','variable','value']] 
df_energy_melted['location'] = df_energy_melted['variable'].str.split('_').str[0] #obs테이블과 조인하기 위해 지역 생성

# df_energy_melted.info()
df_energy_melted = df_energy_melted.dropna()
# df_energy_melted.info()

#주어진 예보정보 전처리
df_fcst_dj = pd.read_csv("data/dangjin_fcst_data.csv")
df_fcst_us = pd.read_csv("data/ulsan_fcst_data.csv")
df_fcst_man = pd.read_csv('data/df_fcst_manual.csv')

#날짜, 시간, Location 정보데이터 추가
df_fcst_dj['date'] = pd.to_datetime(df_fcst_dj['Forecast time'].str.slice(0,10))
df_fcst_dj['hour'] = pd.to_numeric(df_fcst_dj['Forecast time'].str.slice(11,13))

df_fcst_us['date'] = pd.to_datetime(df_fcst_us['Forecast time'].str.slice(0,10))
df_fcst_us['hour'] = pd.to_numeric(df_fcst_us['Forecast time'].str.slice(11,13))
df_fcst_man['date'] = pd.to_datetime(df_fcst_man['date'])

df_fcst_dj['location'] = 'dangjin'
df_fcst_us['location'] = 'ulsan'
df_fcst = pd.concat([df_fcst_dj,df_fcst_us])
df_fcst = pd.merge(df_fcst,df_fcst_man)

#Forecast 시점의 날짜, 시간데이터 추가
df_fcst['fcst_hour'] =  df_fcst['hour'] + df_fcst['forecast']
df_fcst['fcst_date'] =  df_fcst.apply(lambda x : fnc_makeDate(x['date'],x['fcst_hour']) ,axis =1) 
df_fcst['fcst_hour'] =  df_fcst.apply(lambda x : fnc_makeHour(x['fcst_hour']) ,axis =1) 


df_fcst = df_fcst[df_fcst['fcst_hour']<24]
df_fcst = df_fcst[df_fcst['date']!=df_fcst['fcst_date']]

# df_fcst.to_csv('data/df_fcst.csv', index= False)

#예측데이터 평균치 추출
df_fcst[df_fcst['hour']>=20]#최근 예측 데이터만 이용
df_fcst_avg = df_fcst[['location','fcst_date','fcst_hour','Temperature','Humidity','WindSpeed','WindDirection','Cloud','rain','rain_6h','rain_prob','snow_6h']].groupby(['location','fcst_date','fcst_hour'], as_index=False).mean()

# print(df_fcst_avg.head(10))
# print(df_fcst_avg.info())

#빠진 시간 데이터 보간을 위한 데이터 Template 생성
fcst_tmplt = pd.DataFrame()
fcst_tmplt['Forecast_time'] = pd.date_range(start='2018-03-02 00:00:00', end='2021-02-28 23:00:00', freq='H')
fcst_tmplt['fcst_date'] = pd.to_datetime(fcst_tmplt['Forecast_time'].astype(str).str.slice(0,10))
fcst_tmplt['fcst_hour'] = pd.to_numeric(fcst_tmplt['Forecast_time'].astype(str).str.slice(11,13)) 
fcst_tmplt = fcst_tmplt[['fcst_date', 'fcst_hour']]
fcst_tmplt2 = fcst_tmplt.copy()
fcst_tmplt3 = fcst_tmplt.copy()
fcst_tmplt4 = fcst_tmplt.copy()

#location 정보 추가
#사이트 정보 추가
fcst_tmplt['location'] = 'dangjin'
fcst_tmplt['variable'] = 'dangjin'
fcst_tmplt['lat'] = 37.05075279
fcst_tmplt['long'] = 126.5102993

fcst_tmplt2['location'] = 'dangjin'
fcst_tmplt2['variable'] = 'dangjin_warehouse'
fcst_tmplt2['lat'] = 37.05075279
fcst_tmplt2['long'] = 126.5102993

fcst_tmplt3['location'] = 'dangjin'
fcst_tmplt3['variable'] = 'dangjin_floating'
fcst_tmplt3['lat'] = 37.05075279
fcst_tmplt3['long'] = 126.5102993

fcst_tmplt4['location'] = 'ulsan'
fcst_tmplt4['variable'] = 'ulsan'
fcst_tmplt4['lat'] = 35.47765090
fcst_tmplt4['long'] = 129.380778
fcst_tmplt = pd.concat([fcst_tmplt,fcst_tmplt2])
fcst_tmplt = pd.concat([fcst_tmplt,fcst_tmplt3])
fcst_tmplt = pd.concat([fcst_tmplt,fcst_tmplt4])

#빠진시간 데이터 Merge 
print('fcst_tmp len :',len(fcst_tmplt), '/ avg len :', len(df_fcst_avg))
df_fcst_avg = pd.merge(fcst_tmplt,df_fcst_avg, how = 'left')

#데이터 선형 보간
# print(df_fcst_avg.head(20))
df_fcst_avg = df_fcst_avg.interpolate()
# print(df_fcst_avg.head(20))



#모델에 사용할 데이터 생성
df_data = df_energy_melted[['date','hour','variable','location','value']]
df_data.columns = ['fcst_date','fcst_hour','variable','location','value']
df_data = pd.merge(df_data,df_fcst_avg.copy(), how = 'inner')
# df_data = pd.merge(df_fcst_avg,df_data, how = 'left')
# df_data.to_csv('data/df_data.csv')
print(df_data.head(10))


from pysolar.solar import get_altitude, get_azimuth 
from pysolar.radiation import get_radiation_direct
KST = timezone(timedelta(hours=9))
#시간데이터 대신 시간별 태양고도데이터로 대체 
df_data['Forecast_time'] = df_data['fcst_date'].astype(str).str.slice(0,10).map(str) + ' ' + df_data['fcst_hour'].astype(str) +':00'
df_data['altitude'] = df_data.apply(lambda x: get_altitude(x['lat'],x['long'],datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST) ), axis =1)
df_data['altitude'] = df_data['altitude'].apply(lambda x: x if x>=0 else 0)
df_data['angle'] = df_data.apply(lambda x: get_azimuth(x['lat'],x['long'],datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST) ), axis =1)
df_data['radiation'] = df_data.apply(lambda x: get_radiation_direct(datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST),x['altitude']), axis =1)

df_data.to_csv('data/df_data.csv')


#2월 예측을 위한 Test dataset 생성
df_real_test = df_fcst_avg[(df_fcst_avg['fcst_date']>='2021-02-01')&(df_fcst_avg['fcst_date']<='2021-02-28') ].copy()
#시간데이터 대신 시간별 태양고도데이터로 대체 
df_real_test['Forecast_time'] = df_real_test['fcst_date'].astype(str).str.slice(0,10).map(str) + ' ' + df_real_test['fcst_hour'].astype(str) +':00'
df_real_test['altitude'] = df_real_test.apply(lambda x: get_altitude(x['lat'],x['long'],datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST) ), axis =1)
df_real_test['altitude'] = df_real_test['altitude'].apply(lambda x: x if x>=0 else 0)
df_real_test['angle'] = df_real_test.apply(lambda x: get_azimuth(x['lat'],x['long'],datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST) ), axis =1)
df_real_test['radiation'] = df_real_test.apply(lambda x: get_radiation_direct(datetime.strptime(x['Forecast_time'],'%Y-%m-%d %H:%M').replace(tzinfo = KST),x['altitude']), axis =1)
df_real_test.to_csv('data/df_real_test.csv')



