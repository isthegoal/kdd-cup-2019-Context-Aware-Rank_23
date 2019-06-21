# -*- coding: utf-8 -*-
"""
输入文件：feature_692434.csv
          zhu·cc发的：list.npy, dis2bus_test.csv, dis2bus.csv, dis2subway.csv, dis2subway_test.csv, weather_clean.csv

"""
#工具包导入
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import json 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from itertools import product
from sklearn.cluster import KMeans
# 数据读取
path = '../data_set_phase1/'
data = pd.read_csv(path +'feature_692434.csv')

#经纬度聚类将聚类标签作为特征
def cluster_features(data):
    o_co = data[['o']]
    d_co = data[['d']]

    o_co.columns = ['co']
    d_co.columns = ['co']


    data['od_cluster', 'd_cluster'] = np.nan


    all_co = pd.concat([d_co, o_co])['co'].unique()
    X = pd.DataFrame()
    X['lng'] = pd.Series(all_co).apply(lambda x: float(x.split(',')[0]))
    X['lat'] = pd.Series(all_co).apply(lambda x: float(x.split(',')[0]))
    clf_KMeans = KMeans(n_clusters=11)#构造聚类器
    cluster = clf_KMeans.fit_predict(X)#聚类
    index = 0
    for co in tqdm(all_co):
        data.loc[(data['o'] == co), 'o_cluster'] = cluster[index]
        data.loc[(data['d'] == co), 'd_cluster'] = cluster[index]
        index +=1
    return data
data = cluster_features(data)
######################################   非特征    ######################################
or_feature  = ['req_time','click_mode','sid']

######################################   原始特征    ######################################
cate_feature = ['pid'] 
profile_feature = ['p' + str(i) for i in range(66)]

origin_num_feature = ['o_lng', 'o_lat', 'd_lng', 'd_lat'] + profile_feature + cate_feature

######################################   plan统计特征    ######################################
plan_features = ['mode_feas_0','mode_feas_1', 'mode_feas_2', 'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7',
                 'mode_feas_8', 'mode_feas_9', 'mode_feas_10', 'mode_feas_11', 'max_dist', 'min_dist', 'mean_dist', 'std_dist',
                 'max_price', 'min_price', 'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta', 'max_dist_mode',
                 'min_dist_mode', 'max_price_mode', 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'svd_mode_0',
                 'svd_mode_1', 'svd_mode_2', 'svd_mode_3', 'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8', 'svd_mode_9']

######################################   日期特征    ######################################
time_diff_feature=['time_diff']
time_clock_feat=['req_time_hour']
#time_weekday_feat=['req_time_weekday']
time_clock_diff=['diff_6_cloc','diff_12_clock','diff_18_clock','diff_24_clock']
time_jiaocha_detail=[]    #八成降分  去掉了
#holiday_featur     = ['holiday_flag_0','holiday_flag_1','holiday_flag_3']

time_feature=time_diff_feature+time_clock_feat+time_clock_diff +time_jiaocha_detail

######################################   距离特征    ######################################
subway_feature     = ['od_manhattan_distance','o_nearest_dis', 'd_nearest_dis']    #去掉了
distance_center_feature = ['od_manhattan_distance_detail','o_main_centroid_mean_dis','d_main_centroid_mean_dis','o_main_centroid_mode_dis','d_main_centroid_mode_dis']
#bus_distance_feature =  ['o_nearest_bus_dis','d_nearest_bus_dis']  #降分

distance_feature = subway_feature+distance_center_feature

######################################   聚类特征    ######################################
od_features = ['o_cluster','d_cluster']

######################################   编码特征    ######################################
#od_label_encod=['o_label_encoder_all','d_label_encoder_all']   #降分

######################################   平展特征    ######################################
pingzhan_dist_feature=['plan_model_'+str(i)+'_dist'  for i in range(1,12)]
pingzhan_price_feature=['plan_model_'+str(i)+'_price'  for i in range(1,12)]
#pingzhan_eta_feature=['plan_model_'+str(i)+'_eta'  for i in range(1,12)]
pingzhan_rank_feature=['plan_model_'+str(i)+'_rank'  for i in range(1,12)]

pingzhan_feature=pingzhan_dist_feature + pingzhan_price_feature + pingzhan_rank_feature

######################################   统计特征    ######################################
#pid_query_count=['pid_query_count']

statistics_feature= []

######################################   排序特征    ######################################
#位置点出行情况排序
od_apper_rank=['o_appear_count', 'd_appear_count', 'o_appear_count_rank',
       'd_appear_count_rank','o_appear_count_rank_buguiyi', 'd_appear_count_rank_buguiyi']
od_couple_rank=['od_couple_count']

#对自己平展方式效果的排序
pingzhan_dist_rank_feature=['plan_model_'+str(i)+'_dist_rank'  for i in range(1,12)]
pingzhan_price_rank_feature=['plan_model_'+str(i)+'_price_rank'  for i in range(1,12)]
pingzhan_eta_rank_feature=['plan_model_'+str(i)+'_eta_rank'  for i in range(1,12)]
pingzhan_rank_rank_feature=['plan_model_'+str(i)+'_rank_rank'  for i in range(1,12)]
plan_pingzhan_static_rank=pingzhan_dist_rank_feature+pingzhan_price_rank_feature+pingzhan_eta_rank_feature+pingzhan_rank_rank_feature

#对plan统计特征进行排序
#plan_jieshao_static_rank=['max_dist_rank','min_dist_rank','mean_dist_rank','std_dist_rank','max_price_rank','min_price_rank','mean_price_rank','std_price_rank','max_eta_rank','min_eta_rank','mean_eta_rank','std_eta_rank'] 

#对整点时间差进行rank
#time_diff_rank=['diff_6_cloc_rank_buguiyi','diff_12_clock_rank_buguiyi','diff_18_clock_rank_buguiyi','diff_24_clock_rank_buguiyi']

rank_feature=od_apper_rank+od_couple_rank+plan_pingzhan_static_rank

######################################   个人属性特征    ######################################
#prof_svd = ['svd_fea_{}'.format(i) for i in range(20)]

profile=[]

######################################   添加特征    ######################################
        
#计算距离
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1.0 / p)    

def calculate_direction(d_lon, d_lat):  
    result = np.zeros(len(d_lon))   
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])   
    return result

def add_travel_vector_features(df):    
    df['delta_longitude'] = df.o_lng - df.d_lng
    df['delta_latitude'] = df.o_lat - df.d_lat   
    df['pickup_x'] = np.cos(df.o_lat) * np.cos(df.o_lng)
    df['pickup_y'] = np.cos(df.o_lat) * np.sin(df.o_lng)
    df['pickup_z'] = np.sin(df.o_lat)   
    df['dropoff_x'] = np.cos(df.d_lat) * np.cos(df.d_lng)
    df['dropoff_y'] = np.cos(df.d_lat) * np.sin(df.d_lng)
    df['dropoff_z'] = np.sin(df.d_lat)

data['manhattan'] = minkowski_distance(data.o_lng, data.d_lng, 
                                           data.o_lat, data.d_lat, 1)
data['euclidean']=minkowski_distance(data.o_lng, data.d_lng, 
                                           data.o_lat, data.d_lat, 2)

add_travel_vector_features(data)
data['direction'] = calculate_direction(data.delta_longitude, data.delta_latitude)

distance_features = ['manhattan','euclidean','delta_longitude','delta_latitude','pickup_x','pickup_y','pickup_z','dropoff_x','dropoff_y','dropoff_z','direction']

#加大学特征
def school_(x):
    if x*1609<1000:
        return 1
    else:
        return 0

list0 = list(np.load(path+'list.npy'))
list0 = [x*1609 for x in list0]
data["od_s"] = list0

school_feature = ['od_s']

#加公交站台特征
def bus_o(x):
    if x<400:
        return 1
    else:
        return 0
    
def bus_od(x):
    if x<1000:
        return 1
    else:
        return 0
    
dist_bus1 = pd.read_csv( path+'dis2bus_test.csv')
dist_bus2 = pd.read_csv(path+'dis2bus.csv')
dist_bus  = pd.concat([dist_bus1, dist_bus2], ignore_index=True)
data=data.merge(dist_bus, 'left', ['sid'])

data['o_d_dis2bus'] = data['odis2bus']+data['ddis2bus']

bus_feature = ['o_d_dis2bus','odis2bus','ddis2bus']

#加地铁特征
df1 = pd.read_csv( path+ 'dis2subway.csv')
df2 = pd.read_csv( path+ 'dis2subway_test.csv',index_col=0)
df  = pd.concat([df1, df2], ignore_index=0)
data=data.merge(df, 'left', ['sid'])

def subway_o(x):
    if x<600:
        return 1   
    else:
        return 0
    
def subway_od(x):
    if x<1000:
        return 1   
    else:
        return 0

data['o_d_dis2subway']=data['odis2subway']+data['ddis2subway']

subway_feature = ['o_d_dis2subway','odis2subway','ddis2subway']
#加天气特征

#weather特征
weather = pd.read_csv( path+'weather_clean.csv')
weather.rename(columns={'req_time':'req_time_new'},inplace=True) 
weather = weather[['req_time_new','temperature','dewPoint','humidity','humidity','windBearing','windSpeed']]  #weather中的时间是按1小时为时间段
def tran_time(x):   #按照半小时为时间段处理原data中的click_time列
    try:
        if x.minute<30:
            return (x+pd.Timedelta(minutes=-x.minute,seconds = -x.second))    #0~30分钟的按00分算
        else:
            return (x+pd.Timedelta(minutes=-x.minute,hours=1,seconds = -x.second)) #30~60分钟的加一小时
    except:
        return 0

data['req_time_new'] = pd.to_datetime(data['req_time'])

data['req_time_new'] = data['req_time_new'].apply(lambda x:tran_time(x))

#将主key的类型保持一致，不然merge后的weather全为nan
weather["req_time_new"] = weather["req_time_new"].apply(lambda x:str(x))
data["req_time_new"] = data["req_time_new"].apply(lambda x:str(x))
data  = data.merge(weather, 'left', ['req_time_new'])
del data['req_time_new']

weather_feature = ['temperature','dewPoint','humidity','humidity','windBearing','windSpeed']

######################################   特征拼接    ######################################

feature   = or_feature + origin_num_feature + plan_features + time_feature+ distance_feature + od_features + pingzhan_feature + rank_feature + profile + distance_features + school_feature + bus_feature + subway_feature + weather_feature


#删除一部分  低重要度
feature.remove('plan_model_6_price_rank')
feature.remove('plan_model_5_price_rank')
feature.remove('plan_model_5_price')
feature.remove('plan_model_6_price')
feature.remove('plan_model_3_price_rank')
feature.remove('plan_model_3_price')

feature.remove('plan_model_9_rank_rank')
feature.remove('plan_model_6_rank_rank')
feature.remove('plan_model_5_rank_rank')
feature.remove('plan_model_8_rank_rank')
feature.remove('plan_model_10_rank_rank')
feature.remove('plan_model_11_rank_rank')

data1 = data[feature]
feature1 = [col for col in data1.columns if col not in ['req_time','click_mode','sid']]

#模型训练&验证
#评估指标设计
def f1_weighted(labels,preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_weighted', score, True

def pred_proba(proba):
    res = []
#    pred_proba.tolist()
    for i, e in enumerate(proba):
        if e[3] >= 0.180:
            e[3] = 1
        if e[4] >= 0.118:
            e[4] = 1
        if e[6] >= 0.217:
            e[6] = 1
        if e[8] >= 0.21:
            e[8] = 1
        if e[9] >= 0.36:
            e[9] = 1
        if e[10] >= 0.285:
            e[10]=1
        if e[11] >= 0.34:
            e[11]=1
        res.append(e)
    df = pd.DataFrame(res)
    pred = df.idxmax(axis = 1)
    return pred

train_index = (data1.req_time < '2018-11-23') 
train_x     = data1[train_index][feature1].reset_index(drop=True)
train_y     = data1[train_index].click_mode.reset_index(drop=True)

valid_index = (data1.req_time > '2018-11-23') & (data1.req_time < '2018-12-01')
valid_x     = data1[valid_index][feature1].reset_index(drop=True)
valid_y     = data1[valid_index].click_mode.reset_index(drop=True)

test_index = (data1.req_time > '2018-12-01')
test_x     = data1[test_index][feature1].reset_index(drop=True)

print(len(feature1), feature1)
#lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=61, reg_alpha=0, reg_lambda=0.01,
#    max_depth=-1, n_estimators=2000, objective='multiclass',
#    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,min_child_samples = 50,  learning_rate=0.05, random_state=2019, metric="None",n_jobs=-1)
lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=61, reg_alpha=0, reg_lambda=0.01,
    max_depth=-1, n_estimators=2000, objective='multiclass',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,min_child_samples = 50,  learning_rate=0.01, random_state=2019, metric="None",n_jobs=-1)

eval_set = [(valid_x, valid_y)]
lgb_model.fit(train_x, train_y, eval_set=eval_set, eval_metric=f1_weighted, verbose=10, early_stopping_rounds=100)


#特征重要性分析
imp = pd.DataFrame()
imp['fea'] = feature1
imp['imp'] = lgb_model.feature_importances_ 
imp = imp.sort_values('imp',ascending = False)
imp
plt.figure(figsize=[20,10])
sns.barplot(x = 'imp', y ='fea',data = imp.head(20))

#预测结果分析
proba = lgb_model.predict_proba(valid_x)
pred = pred_proba(proba)
score=f1_score(valid_y, pred, average='weighted')
print('offline_f1_score:', score)
df_analysis = pd.DataFrame()
df_analysis['sid']   = data1[valid_index]['sid']
df_analysis['label'] = valid_y.values
df_analysis['pred']  = pred
df_analysis['label'] = df_analysis['label'].astype(int)

score_df = pd.DataFrame(
    columns=['class_id', 'counts*f1_score', 'f1_score', 'precision', 'recall'])

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score
dic_ = df_analysis['label'].value_counts(normalize = True)
def get_weighted_fscore(y_pred, y_true):
    f_score = 0
    for i in range(12):
        yt = y_true == i
        yp = y_pred == i
        f_score += dic_[i] * f1_score(y_true=yt, y_pred= yp)
        score_df.loc[i] = i,dic_[i],f1_score(y_true=yt, y_pred= yp), precision_score(y_true=yt, y_pred= yp),recall_score(y_true=yt, y_pred= yp)
    print('f_score:', f_score)
    return score_df
score_df = get_weighted_fscore(y_true =df_analysis['label'] , y_pred = df_analysis['pred'])
print(score_df)

#模型训练&提交
all_train_x              = data1[data1.req_time < '2018-12-01'][feature1].reset_index(drop=True)
all_train_y              = data1[data1.req_time < '2018-12-01'].click_mode.reset_index(drop=True)
print(lgb_model.best_iteration_)

lgb_model.n_estimators   = lgb_model.best_iteration_
lgb_model.fit(all_train_x, all_train_y)
print('fit over')
result                   = pd.DataFrame()
result['sid']            = data1[test_index]['sid']
result.reset_index(inplace=True)
result.drop(['index'],axis=1,inplace=True)
result_proba = lgb_model.predict_proba(test_x)
a  = pd.DataFrame(pred_proba(result_proba))
result= pd.concat([result,a],axis=1)
result=result.rename(columns={0:'recommend_mode'})
print(len(result))
print(result['recommend_mode'].value_counts())
filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())
result[['sid', 'recommend_mode']].to_csv(filename, index=False)