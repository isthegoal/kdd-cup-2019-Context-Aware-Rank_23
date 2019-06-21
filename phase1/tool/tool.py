import math

#用于根据经纬度获取详细的ob距离
def GetDistance(lng1, lat1, lng2, lat2):
    EARTH_RADIUS = 6378.137

    lng1 = lng1 * math.pi / 180.0
    lng2 = lng2 * math.pi / 180.0
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    dis1 = lat1 - lat2
    dis2 = lng1 - lng2

    s = 2 * math.asin(
        ((math.sin(dis1 / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dis2 / 2)) ** 2) ** 0.5)
    s = s * EARTH_RADIUS * 1000
    return s