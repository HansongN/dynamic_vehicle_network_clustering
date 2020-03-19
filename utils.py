# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/10 21:20 

import math
from math import sin, cos, pi, radians, fabs, asin, sqrt
import time
import json
import pickle
import numpy as np

EARTH_REDIUS = 6378.137

def rad(d):
    return d * pi / 180.0

def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_REDIUS * asin(sqrt(h))

    return distance


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_dict(filepath, mode, dic):
    out = open(filepath, mode)
    out.write(json.dumps(dic, cls=MyEncoder) + "\n")
    out.close()


def load_dict(filepath):
    with open(filepath, "r") as lines:
        for line in lines:
            line_json = json.loads(line)
    lines.close()
    return line_json

def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open(path, 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


 # 100 0.4/400=0.001
if __name__ == '__main__':
    s = getDistance(30.07, 120.0, 30.07, 120.001)
    print(s)
