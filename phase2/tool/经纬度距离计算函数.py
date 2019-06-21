# -*- coding: utf-8 -*-
import math
# #2、直线距离特征：根据经纬度坐标计算两点间的直线距离
# def GetDistance(lng1, lat1, lng2, lat2):
#     #地球赤道半径6378.137km
#     EARTH_RADIUS = 6378.137
#     #把经纬度转换成度（°）
#     lng1 = lng1 * (math.pi / 180.0)
#     lng2 = lng2 * (math.pi / 180.0)
#     lat1 = lat1 * (math.pi / 180.0)
#     lat2 = lat2 * (math.pi / 180.0)
#
#     dis1 = lat1 - lat2
#     dis2 = lng1 - lng2
#     #以1m为球的半径，求球上两点的距离
#     s = 2 * math.asin(
#         ((math.sin(dis1 / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dis2 / 2)) ** 2) ** 0.5)
#     #换算成地球的半径：6378.137km *1000 = 6378137m
#     s = s * EARTH_RADIUS * 1000
#     return s
#
#
# o_lng=116.42
# o_lat=40.0
# d_lng=116.423128
# d_lat=40.001439
# print(GetDistance(o_lng,o_lat,d_lng,d_lat))

a='<lat>39.891486</lat>'
print(a[5:-6])