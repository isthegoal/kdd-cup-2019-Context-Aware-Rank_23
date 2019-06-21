# coding=gbk
'''

    爬取的地址：http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-placeapi
    爬取用的key值为：LlNVsWOsqmKwlqkUEBmzbpmbu2GAc2Dl
    http://api.map.baidu.com/place/v2/search?query=ATM机&tag=银行&region=北京&output=json&ak=LlNVsWOsqmKwlqkUEBmzbpmbu2GAc2Dl
'''

import pandas as pd
import urllib.request
import json
import re
import time


http='http://api.map.baidu.com/geocoder/v2/?callback=renderReverse&location=35.658651,139.745415&output=json&pois=1&latest_admin=1&ak=hCedZFZlgn8lOvvOId5hrXdvDZVQDrKr'

# def gen_all_address():
#     data=pd.read_csv('../data_set_phase1/feature_data/feature.csv')
#     # the_orig_lng=data['o_lng']
#     # the_orig_lat=data['o_lat']
#     # the_des_lng=data['d_lng']
#     # the_des_lat=data['d_lat']
#     print(2)
#     the_quyu=[]
#     print(len(data.index))
#     for indexs in data.index:
#         print(len(the_quyu))
#         resp_orig = urllib.request.urlopen('http://api.map.baidu.com/geocoder/v2/?callback=renderReverse&location='+str(data.loc[indexs]['o_lat'])+','+str(data.loc[indexs]['o_lng'])+'&output=json&pois=1&latest_admin=1&ak=hCedZFZlgn8lOvvOId5hrXdvDZVQDrKr')
#         resp_str=str(resp_orig.read().decode('utf8'))
#         resp_str=resp_str[29:-2]
#
#
#         p1 = r"北京市.*区"
#         pattern1 = re.compile(p1)
#
#         the_quyu.append(pattern1.findall(resp_str)[0].split('北京市')[1].split('区')[0])
#
#         time.sleep(1)
#         #############   使用正则表达式进行解析提取   ##############
#     print(the_quyu)
#
#     return the_quyu
#
# def test_zhengze():
#     my_txt='{"status":0,"result":{"location":{"lng":116.28999999999994,"lat":39.97000007254172},"formatted_address":"北京市海淀区蓝靛厂中路","business":"世纪城,四季青,远大路","addressComponent":{"country":"中国","country_code":0,"country_code_iso":"CHN","country_code_iso2":"CN","province":"北京市","city":"北京市","city_level":2,"district":"海淀区","town":"","adcode":"110108","street":"蓝靛厂中路","street_number":"","direction":"","distance":""},"pois":[{"addr":"北京市海淀区蓝靛厂居住区世纪城3期春荫园6号楼","cp":" ","direction":"北","distance":"102","name":"中国建设银行(北京远大中路支行)","poiType":"金融","point":{"x":116.28983435050277,"y":39.96930388943203},"tag":"金融;银行","tel":"","uid":"d4034eb38a6c2ff6c0364441","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"蓝靛厂中路19号","cp":" ","direction":"西南","distance":"203","name":"蓝靛厂清真寺","poiType":"旅游景点","point":{"x":116.29121774098813,"y":39.971045708291629},"tag":"旅游景点;教堂","tel":"","uid":"5191bb9a6696551b1ce99987","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"北京市海淀区蓝靛厂春荫园小区5号楼","cp":" ","direction":"东北","distance":"233","name":"蓝靛厂春荫园","poiType":"房地产","point":{"x":116.2883162141909,"y":39.969034318229848},"tag":"房地产;住宅区","tel":"","uid":"34f68ab038f0530a8e55ecac","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"北京市海淀区蓝靛厂翠叠园小区9号楼","cp":" ","direction":"西北","distance":"250","name":"蓝靛厂翠叠园","poiType":"房地产","point":{"x":116.29195435150632,"y":39.96913799958826},"tag":"房地产;住宅区","tel":"","uid":"e93761f6bb9d9572223bf270","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"蓝晴路与蓝靛厂中路交叉口西北150米","cp":" ","direction":"东南","distance":"161","name":"曙光街道温馨家园","poiType":"房地产","point":{"x":116.28907977387439,"y":39.970859086982319},"tag":"房地产;住宅区","tel":"","uid":"7bcfc17f6fa9933e077ce8aa","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"海淀区长春桥北300米","cp":" ","direction":"南","distance":"216","name":"世纪城(三期)","poiType":"房地产","point":{"x":116.29049909658015,"y":39.97144659677496},"tag":"房地产;住宅区","tel":"","uid":"75fa23f1956124bed3ff23bb","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"海淀区蓝靛厂中路垂虹园甲4号楼","cp":" ","direction":"北","distance":"378","name":"中国民生银行(北京世纪金源支行)","poiType":"金融","point":{"x":116.29070570684745,"y":39.96743760529318},"tag":"金融;银行","tel":"","uid":"861a8cf81cd8b28dfb2d6117","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"北京市海淀区蓝靛厂路17号","cp":" ","direction":"南","distance":"440","name":"蓝靛清云五金建材商店","poiType":"购物","point":{"x":116.28925943497639,"y":39.972987921801209},"tag":"购物;家居建材","tel":"","uid":"61fa14618082975d3a04b1ea","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"北京市海淀区晴波园甲1号","cp":" ","direction":"西","distance":"393","name":"车享家(晴波园店)","poiType":"汽车服务","point":{"x":116.29353536920388,"y":39.97005038870725},"tag":"汽车服务;汽车维修","tel":"","uid":"af1abb69134d84ee92114c59","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}},{"addr":"北京市海淀区老营房路春荫园12号楼3-4单元1层","cp":" ","direction":"东北","distance":"390","name":"红果果儿童之家(幼儿园)","poiType":"教育培训","point":{"x":116.28750773923193,"y":39.96810117887099},"tag":"教育培训;幼儿园","tel":"","uid":"80a73b12e198d790774ff1fc","zip":"","parent_poi":{"name":"","tag":"","addr":"","point":{"x":0.0,"y":0.0},"direction":"","distance":"","uid":""}}],"roads":[],"poiRegions":[],"sematic_description":"中国建设银行(北京远大中路支行)北102米","cityCode":131}}'
#     p1 = r"北京市.*区"
#     pattern1 = re.compile(p1)
#
#     print(pattern1.findall(my_txt)[0].split('北京市')[1].split('区')[0])
#





def gen_beijing_external_data(coor):
    all_point=coor['o']|coor['d']
    print(all_point)
    return 0


def gen_shanghai_external_data(coor):
    return 0


def gen_shenzhen_external_data(coor):
    return 0



beijing_feature=pd.read_csv('../feature_data/feature_beijing.csv')
gen_beijing_external_data(beijing_feature[['o','d']])

shanghai_feature=pd.read_csv('../feature_data/feature_shanghai.csv')
gen_shanghai_external_data(shanghai_feature[['o','d']])

shenzhen_feature=pd.read_csv('../feature_data/feature_shenzhen.csv')
gen_shenzhen_external_data(shenzhen_feature[['o','d']])