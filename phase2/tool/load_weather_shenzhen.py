import forecastio #module to get weather data from darksky.net
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from calendar import monthrange
import os.path
import datetime
'''
  主要由经纬度保证爬取天气的地方。
'''
api_key = '276d9b4ae748ec5d42ab2ababe8435cc' #apikey obtained from darksky.net
#api_key = '999653ec682af395046067847b4f4948' #apikey obtained from darksky.net by Yash
#api_key = '40a186e05ecf67b4dfd467472fdd35bb' #another apikey obtained from darksky.net by Yash
#api_key='12defc9d5eb6b4592d9b043f29344c1e'
#api_key='4d83264f9aa9a9e9acf918e8ad53313a'
#api_key='d516390e57da4a682a3c8b6ec267b542'


def get_met_data(start_date, numdays, api_key, lat, lng, station_id):
    "Function to get weather"

    #get url to retrieve weather information
    date_list = [start_date + datetime.timedelta(days = x) for x in range(0, numdays)]
    hist = np.arange(0, len(date_list)).tolist()
    forecast = []
    for n in hist:
        forecast.append(forecastio.load_forecast(api_key, lat, lng, date_list[n]))
    #jspn object
    met_data = pd.DataFrame()
    for i in np.arange(0, len(forecast)).tolist():
        data = json_normalize(forecast[i].json['hourly']['data'])
        met_data = pd.concat([met_data, data])
    #missing variables from original data added
    met_data['datetime'] = met_data['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    met_data['lat'] = lat
    met_data['long'] = lng
    met_data['station_id'] = station_id
    return met_data

def getWeatherDataRange(startDate, endDate, stationsNeeded, cityName, shortRun = True):
    #设置自动化的爬取四个时间结点段和  key的替换
    the_start_year=0
    the_start_month=0
    the_start_day=0
    the_end_year=0
    the_end_month=0
    the_end_day=0
    for i in [4]:
        print(i)
        if i==0:
            api_key = '276d9b4ae748ec5d42ab2ababe8435cc'
            the_start_year = 2018
            the_start_month = 9
            the_start_day = 28
            the_end_year = 2018
            the_end_month = 10
            the_end_day = 2
        if i==1:
            api_key = '999653ec682af395046067847b4f4948'
            the_start_year = 2018
            the_start_month = 10
            the_start_day = 3
            the_end_year = 2018
            the_end_month = 10
            the_end_day = 19

        if i==2:
            api_key = '40a186e05ecf67b4dfd467472fdd35bb'
            the_start_year = 2018
            the_start_month = 10
            the_start_day = 20
            the_end_year = 2018
            the_end_month = 11
            the_end_day = 6
        if i==3:
            api_key = '4d83264f9aa9a9e9acf918e8ad53313a'
            the_start_year = 2018
            the_start_month = 11
            the_start_day = 7
            the_end_year = 2018
            the_end_month = 11
            the_end_day = 22
        if i==4:
            api_key ='d516390e57da4a682a3c8b6ec267b542'
            the_start_year = 2018
            the_start_month = 11
            the_start_day = 23
            the_end_year = 2018
            the_end_month = 12
            the_end_day = 7



        startDate = datetime.datetime.strptime(str(the_start_year) + '-' + str(the_start_month) + '-' + str(the_start_day),
                                             '%Y-%m-%d')
        endDate = datetime.datetime.strptime(str(the_end_year) + '-' + str(the_end_month) + '-' + str(the_end_day),
                                             '%Y-%m-%d')



        print('在得到天气时候，  开始是:',startDate,'   结束是：',endDate)

        "Retrieves the weather data from file or from web"
        assert(startDate<endDate)
        if os.path.exists(u'E:/Machine-learning/kdd_cup_2018/DS420-Cobras-master/viz/' + cityName + '_weather.csv'):
            cached =  pd.read_csv(u'E:/Machine-learning/kdd_cup_2018/DS420-Cobras-master/viz/' + cityName + '_weather.csv')
        else:
            cached = pd.DataFrame()
        if 'datetime' in set(cached):
            cached['datetime'] = pd.to_datetime(cached['datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

        alreadyHave = {}

        datesNeeded = []
        curDate =  startDate
        while curDate <= endDate:
            #无间隔的获取所有的时间点，之前还是有坑的，这样获取下来  所有的时间点都已经包含了，（只不过要每个站点都重新爬去）
            datesNeeded.append(curDate)
            curDate = curDate + datetime.timedelta(hours=1)
        count = 0
        print(datesNeeded)
        for stations in stationsNeeded:
            # if count == 0 and shortRun:
            #     break

            #print(str(stationInfo['station_id'].values[0]))
            prevCount = count
            for dates in datesNeeded:
                print('获取天气时候  需要的时间点数据为：', dates)
                #if count == 3 and shortRun:
                #    break
                if (stations, dates) not in alreadyHave:
                    count += 1
                    #print('  加1了把')

                    cached = pd.concat([cached, get_met_data(dates, 1, api_key, float(22.5386), float(114.0592), str(stations) )])

                    print('the  cached is:' ,get_met_data(dates, 1, api_key, float(22.5386), float(114.0592), str(stations)))

            for i in range(5): # Try 5 times
                cached.to_csv('E:/Machine-learning/kdd_cup_2018/DS420-Cobras-master/cache.csv')
                cached = cached[np.logical_not(cached.duplicated())] # Remove duplicates if they somehow make it into the system
                #cached.to_csv('/home/fly/Desktop/cache1.csv')
                cached = cached[np.logical_not(cached[['station_id', 'datetime']].duplicated())]  #这步很邪恶，把我需要的信息弄没了  这是个坑，把time改成datatime就好了，草，之前都不行，哀，那个人故意的
                #cached.to_csv('/home/fly/Desktop/cache2.csv')
                if cached['datetime'].dtype == np.dtype('O'):
                    cached['datetime'] = pd.to_datetime(cached['datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
                if count != prevCount: # Only write to file if we loaded something from internet
                    #print('写进去阿，  这里的cached是：',cached)
                    try:
                        cached.to_csv(os.path.join('viz', cityName + '_weather.csv'), index=False)
                    except:
                        if i == 4:
                            raise
                    else:
                        break
    return cached




#一次19天，采集5次 haidian.meo作为最后的天气。  10.1-10.18   10.19-11.6   11.7-11.24  11.25-12.12 12.13-12.29
getWeatherDataRange('', '', ['Shenzhenshi_meo'], 'Shenzhenshi', shortRun = True)




