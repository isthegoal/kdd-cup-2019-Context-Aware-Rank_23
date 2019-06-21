# coding=gbk
import pandas as pd

def gen_info():
    subway_info=pd.read_csv('../data/beijing subway.csv',header=None,index_col=0)
    subway_info.rename(columns={3:'address'},inplace=True)

    subway_info['station_latitude']=subway_info['address'].apply(lambda x:x.split(', ')[0])
    subway_info['station_longitude']=subway_info['address'].apply(lambda x:x.split(', ')[1])
    print(subway_info.head())
    subway_info.to_csv('../data/beijing_subway.csv')

gen_info()