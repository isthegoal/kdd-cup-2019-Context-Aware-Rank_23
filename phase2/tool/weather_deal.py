
import pandas as pd
import numpy as np
def deal_shenzhen_weather():
    weather_shenzhen=pd.read_csv('E:/Machine-learning/kdd_cup_2019/phase2/data/use_dataset/weather_shenzhen.csv',parse_dates=['datetime'])
    print(np.max(weather_shenzhen['datetime']))
    weather_shenzhen.sort_values(by=['datetime'])
    weather_shenzhen.drop_duplicates(inplace=True)
    weather_shenzhen.to_csv('E:/Machine-learning/kdd_cup_2019/phase2/data/use_dataset/weather_shenzhen_after_deal.csv')
if __name__=='__main__':
    deal_shenzhen_weather()