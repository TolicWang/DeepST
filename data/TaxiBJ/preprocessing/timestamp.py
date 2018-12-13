# @Time    : 2018/12/5 9:28
# @Email  : wangchengo@126.com
# @File   : timestamp.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import time
import pandas as pd
import numpy as np
from datetime import datetime


def string2timestamp(strings, T=48):
    """
    将字符串类型的时间转换成时间戳格式
    :param strings:
    :param T:
    :return:
    example:
    str = [b'2013070101', b'2013070102']
    print(string2timestamp(str))
    [Timestamp('2013-07-01 00:00:00'), Timestamp('2013-07-01 00:30:00')]
    """
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


def timestamp2vec(timestamps):
    """
    将时间戳转换为表示星期几和工作日的向量
    :param timestamps:
    :return:
    exampel:
    [b'2018120505', b'2018120106']
    #[[0 0 1 0 0 0 0 1]  当天是星期三，且为工作日
     [0 0 0 0 0 1 0 0]]  当天是星期六，且为休息日

    """
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8],encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    # vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


if __name__ == "__main__":
    # t = ['2013-06-30 23:30:00']#
    t= [b'2018120505', b'2018120106']
    print(timestamp2vec(t))
    print([0 for _ in range(7)])
