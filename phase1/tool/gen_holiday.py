# coding=gbk
import pandas as pd

server_url = "http://api.goseek.cn/Tools/holiday?date="
import urllib.request
import json
import datetime
def crawel_holiday():
    '''
    现在已经有20170101  到 20181220 的所有日期标识信息
    其中正常工作日对应结果为 0, 法定节假日对应结果为 1, 节假日调休补班对应的结果为 2，休息日对应结果为 3
    '''
    start_data='2018-06-01 19:00:00'
    end_data = '2018-12-20 19:00:00'
    start_timestamp = pd.to_datetime(start_data)
    end_timestamp = pd.to_datetime(end_data)
    have_holiday=pd.read_csv('../data/holiday_bj.csv')
    have_holiday.pop('lunardate') #不要农历
    print('----    开始循环捕获日期    ----')
    while start_timestamp<=end_timestamp:
        i=str(start_timestamp)
        resp = urllib.request.urlopen(server_url + str(i[0:4]) + str(i[5:7]) + str(i[8:10]))
        html = json.loads(resp.read())
        date_flag = html['data']
        print('the date_flag:',date_flag)

        new = pd.DataFrame({"date": str(i[0:4]) + str(i[5:7]) + str(i[8:10]),  "holiday": date_flag}, index=["0"])
        have_holiday =have_holiday.append(new,ignore_index=True)
        have_holiday.to_csv('../data/holiday_china_all.csv')
        start_timestamp = start_timestamp + datetime.timedelta(hours=24)

crawel_holiday()