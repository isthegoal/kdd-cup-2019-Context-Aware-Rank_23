# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:32:00 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import os

import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import json 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from itertools import product
import ast
import math
from sklearn.cluster import KMeans
import time
from datetime import datetime

######################################## 基本数据集加载  ########################################
data_path = '../data_set_phase2/'
print('基本数据集加载')
train_queries1 = pd.read_csv(data_path + 'train_queries_phase1.csv', parse_dates=['req_time'])
train_plans1   = pd.read_csv(data_path + 'train_plans_phase1.csv', parse_dates=['plan_time'])
train_clicks1  = pd.read_csv(data_path + 'train_clicks_phase1.csv')

train_queries2 = pd.read_csv(data_path + 'train_queries_phase2.csv', parse_dates=['req_time'])
train_plans2   = pd.read_csv(data_path + 'train_plans_phase2.csv', parse_dates=['plan_time'])
train_clicks2  = pd.read_csv(data_path + 'train_clicks_phase2.csv')

profiles      = pd.read_csv(data_path + 'profiles.csv') 
test_queries  = pd.read_csv(data_path + 'test_queries.csv', parse_dates=['req_time'])
test_plans    = pd.read_csv(data_path + 'test_plans.csv', parse_dates=['plan_time'])
#train_queries1:(500000,5)
#train_plans1:(491054,3)
#train_clicks1:(453336,3)
#train_queries2:(1500000,5)
#train_plans2:(1447518,3)
#train_clicks2:(1221847,3)
#profiles:(119856,67)
#test_queries:(304916,5)
#test_plans:(296483,3)
print('完成加载')

######################################## 基本数据集合并  ########################################
print('基本数据集合并')
train_queries = pd.concat([train_queries1, train_queries2], ignore_index=True)
train_plans   = pd.concat([train_plans1, train_plans2], ignore_index=True)
train_clicks  = pd.concat([train_clicks1, train_clicks2], ignore_index=True)

train = train_queries.merge(train_plans, 'left', ['sid'])
test  = test_queries.merge(test_plans, 'left', ['sid'])
train = train.merge(train_clicks, 'left', ['sid'])
train['click_mode'] = train['click_mode'].fillna(0).astype(int)
data  = pd.concat([train, test], ignore_index=True)
data  = data.merge(profiles, 'left', ['pid']) 
#train_queries:(2000000,5)
#train_plans:(1938572,3)
#train_clicks:(1675183,3)
#train:(2000000,9)
#test:(304916,7)
#data:(2304916,75)
print('完成合并')

######################################## 提取test_sid ########################################
print('提取test_sid')
test_index = (data.req_time > '2018-12-01')
test_sid  = pd.DataFrame()
test_sid['sid'] = data[test_index]['sid']
test_sid.reset_index(inplace=True)
test_sid.drop(['index'],axis=1,inplace=True)
test_sid['sid'].to_csv('test_sid.csv', index=False,header=1)

######################################## 地点分离  ########################################
print('地点分离')
data['o_lng'] = data['o'].apply(lambda x: float(x.split(',')[0]))
data['o_lat'] = data['o'].apply(lambda x: float(x.split(',')[1]))
data['d_lng'] = data['d'].apply(lambda x: float(x.split(',')[0]))
data['d_lat'] = data['d'].apply(lambda x: float(x.split(',')[1])) 
kmeans = KMeans(n_clusters=3)
data.loc[:, 'o_kmeans'] = kmeans.fit_predict(data[['o_lng', 'o_lat']])
label_pred = kmeans.labels_
data_shenzhen = data[data.o_kmeans == 0]
data_beijing = data[data.o_kmeans == 1]
data_shanghai = data[data.o_kmeans == 2]
#绘图
plt.figure()
plt.scatter(data_shenzhen['o_lng'], data_shenzhen['o_lat'], c = "red", marker='o', label='shenzhen')
plt.scatter(data_beijing['o_lng'], data_beijing['o_lat'], c = "green", marker='*', label='beijing')
plt.scatter(data_shanghai['o_lng'], data_shanghai['o_lat'], c = "blue", marker='+', label='shanghai')
plt.xlabel('o_lng')
plt.ylabel('o_lat')
plt.legend()
plt.show()

#删除
del data_shenzhen['o_lng'],data_shenzhen['o_lat'],data_shenzhen['d_lng'],data_shenzhen['d_lat'],data_shenzhen['o_kmeans']
del data_beijing['o_lng'],data_beijing['o_lat'],data_beijing['d_lng'],data_beijing['d_lat'],data_beijing['o_kmeans']
del data_shanghai['o_lng'],data_shanghai['o_lat'],data_shanghai['d_lng'],data_shanghai['d_lat'],data_shanghai['o_kmeans']

print(len(data_beijing))
#分开保存
data_shenzhen.to_csv("data_shenzhen.csv", index=False)
data_beijing.to_csv("data_beijing.csv", index=False)
data_shanghai.to_csv("data_shanghai.csv", index=False)

print('完成地点分离')

