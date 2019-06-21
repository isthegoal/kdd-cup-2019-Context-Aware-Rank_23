# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
address_all_info_area=pd.read_csv('./address_all_info_area.csv')
print(address_all_info_area)

#2、直线距离特征：根据经纬度坐标计算两点间的直线距离
# 引入地铁站数据
subwayinfo = pd.read_csv('../data/beijing_subway.csv')


# 构建真实距离 的最小值
def GetDistance_min(lat1, lng1):
    # print(1)
    lng2 = address_all_info_area['address_lng']
    lat2 = address_all_info_area['address_lat']

    EARTH_RADIUS = 6378.137

    lng1 = lng1 * math.pi / 180.0
    lng2 = lng2 * math.pi / 180.0
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    s_list = []

    for i in range(len(lng2)):
        dis1 = lat1 - lat2[i]
        dis2 = lng1 - lng2[i]
        s = 2 * math.asin(
            ((math.sin(dis1 / 2)) ** 2 + math.cos(lat1) * math.cos(lat2[i]) * (math.sin(dis2 / 2)) ** 2) ** 0.5)
        s = s * EARTH_RADIUS * 1000
        s_list.append(s)

    index_min = np.argsort(s_list)
    # print(s_list[index_min[0]])
    # print(s_list[index_min[1]])
    if s_list[index_min[1]] < 2000:
        if address_all_info_area['Residential_area_list'][index_min[1]] != '不可知':

            return address_all_info_area['Residential_area_list'][index_min[1]]
        else:
            return '附近位置搜索，失败'
    else:
        return '附近位置搜索，失败'


def use_to_get_near_area(address_lat, address_lng):
    temp = GetDistance_min(address_lat, address_lng)
    return temp


def area_hou_deal(area_str):
    # print('一次',area_str['address_lng'])
    if len(area_str['Residential_area_list']) > 13:
        # print('mark下 直接返回')
        new_area = area_str['Residential_area_list']

    elif len(area_str['Residential_area_list']) == 13:
        # print('mark下 需要处理1')
        new_area = use_to_get_near_area(area_str['address_lat'], area_str['address_lng'])
        # print(new_area)
    elif len(area_str['Residential_area_list']) < 13:
        # print('mark下 需要处理2')
        new_area = use_to_get_near_area(area_str['address_lat'], area_str['address_lng'])
        # print(new_area)
    # print(new_area)
    return new_area


address_all_info_area['after_deal_area'] = address_all_info_area[
    ['Residential_area_list', 'address_lat', 'address_lng']].apply(area_hou_deal, axis=1)

address_all_info_area.to_csv('./address_all_info_area_fill.csv')
