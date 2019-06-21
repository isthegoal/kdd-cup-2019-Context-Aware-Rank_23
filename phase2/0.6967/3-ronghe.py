# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:32:00 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime

######################################## 基本数据集加载  ########################################
data_path = '../ronghe_data/'
data = pd.read_csv(data_path + 'test_sid.csv')
shenzhen = pd.read_csv(data_path + 'shenzhen_2019-06-08_22_56_sub.csv')
beijing = pd.read_csv(data_path + 'beijing_2019-06-06_13_36_sub.csv')
shanghai = pd.read_csv(data_path + 'shanghai_2019-06-06_15_48_sub.csv')
All  = pd.concat([shenzhen, beijing, shanghai], ignore_index=True, axis = 0)
data  = data.merge(All, 'left', ['sid'])
filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())
data.to_csv(filename, index=False)

