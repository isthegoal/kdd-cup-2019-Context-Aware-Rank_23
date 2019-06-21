
#pandas下没有直接多进程的方法，这里是将  原始apply改成 group 之后的方式实现，可用于提取特征缩短时间。   可以发现主要是多添加了两个函数
'''

对应于下式，原始的形式为：
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

def add_labels(filenam,df):
	list_name = list(df['name'])
	if filename in list_name:
		i = list_name.index(filename)
		return df['是否购买][i]
	else:
		return 'Nan'

df1['是否购买'] = df1['name'].apply(add_labels, args=(df2,))
'''
import pandas as pd
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

from joblib import Parallel, delayed
def add_labels(filename, df):
    list_name = list(df['name'])
    if filename in list_name:
        i = list_name.index(filename)
        return df['是否购买'][i]
    else:
        return 'Nan'


def tmp_func(df1):
    df1['是否购买'] = df1['name'].apply(add_labels, args=(df2,))
    return df1


def apply_parallel(df_grouped, func):
    results = Parallel(n_jobs=10)(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(results)


df_grouped = df1.groupby(df1.index)
df1 = apply_parallel(df_grouped, tmp_func)
