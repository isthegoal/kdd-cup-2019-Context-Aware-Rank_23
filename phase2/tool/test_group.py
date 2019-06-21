# coding=gbk
import pandas as pd
import numpy as np

np.random.seed(1)

""" 1. groupby, 按键拆分, 重组, 求和 """
df = pd.DataFrame({
    "key1": ["a", "a", "b", "b", "a"],
    "key2": ["one", "two", "one", "two", "one"],
    "data1": [np.nan,2,3,4,5],
    "data2": [6,7,8,9,10]
})

# 按key1分组, 计算data1列的平均值
key1 = df.groupby(df["key1"])['data1'].count()

print(df)
print(key1)
