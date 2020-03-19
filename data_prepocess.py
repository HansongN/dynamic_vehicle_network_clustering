# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/10/10 21:25 

import csv
import time
import os

"""
['REC_CARID', 'REC_TIME', 'REC_LONGITUDE', 'REC_LATITUDE', 
'REC_SPEED', 'REC_DIRECTION', 'REC_CARSTATUS', 'DBTIME']
"""

if __name__ == '__main__':
    filename = r"raw_data\data1.txt"

    # hour_times = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
    #               10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
    #               20: 0, 21: 0, 22: 0, 23: 0}
    # # long = []
    # # lati = []
    # cid_hour_times = dict()
    # with open(r"E:\Dataset\杭州出租车数据备份\data1.txt", "r") as lines:
    #     for line in lines:
    #         a = line.split()
    #         if a[3] not in ["0", "3"] and a[4] != "0-0-0":
    #             # long.append(a[1])
    #             # lati.append(a[2])
    #             # cid.add(a[0])
    #             if a[0] not in cid_hour_times:
    #                 cid_hour_times[a[0]] = hour_times
    #             #     gps_time = a[4] + " " + a[5]
    #             #     hour = time.strptime('gps_time', '%Y-%m-%d %H:%M:%S').tm_hour
    #             #     cid_hour_times[a[0]][hour] += 1
    #             # else:
    #             gps_time = a[4] + " " + a[5]
    #             # print(a[0], gps_time)
    #             hour = time.strptime(gps_time, '%Y-%m-%d %H:%M:%S').tm_hour
    #             cid_hour_times[a[0]][hour] += 1
    # lines.close()
    # cids = set()
    # for cid, hour_times in cid_hour_times.items():
    #     times = list(hour_times.values())
    #     print(times)
    #     if 0 not in times:
    #         cids.add(cid)
    # print(len(cids))

    # 30.27 120.2

    # output = open(r"handled_data\data1.txt", "w")
    # with open(r"raw_data\data1.txt", "r") as lines:
    #     for line in lines:
    #         a = line.split()
    #         if a[3] not in ["0", "3"] and a[4] != "0-0-0" and float(a[1])>=120.0 and float(a[1])<=120.4 and float(a[2])>=30.07 and float(a[2])<=30.47:
    #             output.write(line)
    # lines.close()
    # output.close()

    # output = open(r"handled_data\data(0-1).txt", "w")
    # with open(r"handled_data\data1.txt", "r") as lines:
    #     for line in lines:
    #         a = line.split()
    #         gps_time = a[4] + " " + a[5]
    #         hour = time.strptime(gps_time, '%Y-%m-%d %H:%M:%S').tm_hour
    #         output_filename = "data1" + "(" + str(hour) + "-" + str(hour+1) + ").txt"
    #         output = open(os.path.join("handled_data", output_filename), "a")
    #         output.write(line)
    #         output.close()
    # lines.close()

    # for i in range(1, 24):
    #     output_filename = "data1\data1" + "(" + str(i) + "-" + str(i + 1) + ").txt"
    #     output = open(output_filename, "w")
    #     with open(r"handled_data\data1\data1.txt", "r") as lines:
    #         for line in lines:
    #             a = line.split()
    #             gps_time = a[4] + " " + a[5]
    #             hour = time.strptime(gps_time, '%Y-%m-%d %H:%M:%S').tm_hour
    #             if hour == i:
    #                 output.write(line)
    #
    #     lines.close()
    #     output.close()

    for i in range(8, 21):
        filename = r"handled_data\data1\byHour\data1(" + str(i) + "-" + str(i + 1) + ").txt"
        for j in [10, 30, 40, 50]:
            cids = list()
            output_filename = r"handled_data\data1\byMinute\data1_" + str(i) + "_" + str(j) + ".txt"
            output = open(output_filename, "w")
            with open(filename, "r") as lines:
                for line in lines:
                    a = line.split()
                    gps_time = a[4] + " " + a[5]
                    min = time.strptime(gps_time, '%Y-%m-%d %H:%M:%S').tm_min
                    if min == j and a[0] not in cids:
                        cids.append(a[0])
                        output.write(line)
            lines.close()
            output.close()

    # with open(r"handled_data\data1\byMinute\data1_8_0.txt", "r") as lines:
    #     for line in lines:
    #         line_json = json.loa

