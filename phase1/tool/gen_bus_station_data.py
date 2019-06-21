# coding=gbk
import pandas as pd
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import seaborn as sns
import matplotlib.pyplot as plt
point_station=pd.read_table('../data/beijing_bus_station.txt',sep=',')


#根据经纬度进行站点的位置绘制，看是否计算够真实

#plt.figure(figsize=(30, 30))
point_station.rename(columns={'x':'lng','y':'lat'},inplace=True)
point_station.plot.scatter(x='lng', y='lat',marker='.')
point_station.pop('FID')
print(point_station)

point_station.to_csv('../data/beijing_bus_station.csv')
plt.show()
