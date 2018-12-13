# @Time    : 2018/12/4 14:57
# @Email  : wangchengo@126.com
# @File   : STMatrix.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import numpy as np
import pandas as pd
from .timestamp import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps# [b'2013070101', b'2013070102']
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()  # 将时间戳：做成一个字典，也就是给每个时间戳一个序号

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):  # 给定时间戳返回对于的数据
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version

        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 时间偏移 minutes = 30
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        # print depends # [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            # 取当前时刻的前3个时间片的数据数据构成“邻近性”模块中一个输入序列
            # 例如当前时刻为[Timestamp('2013-07-01 00:00:00')]
            # 则取：
            # [Timestamp('2013-06-30 23:30:00'), Timestamp('2013-06-30 23:00:00'), Timestamp('2013-06-30 22:30:00')]
            #  三个时刻所对应的in-out flow为一个序列
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            # 取当前时刻 前 1*PeriodInterval,2*PeriodInterval,...,len_period*PeriodInterval
            # 天对应时刻的in-out flow 作为一个序列，例如按默认值为 取前1、2、3天同一时刻的In-out flow
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            # 取当前时刻 前 1*TrendInterval,2*TrendInterval,...,len_trend*TrendInterval
            # 天对应时刻的in-out flow 作为一个序列,例如按默认值为 取 前7、14、21天同一时刻的In-out flow
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
                # a.shape=[2,32,32] b.shape=[2,32,32] c=np.vstack((a,b)) -->c.shape = [4,32,32]
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])#[]
            i += 1
        XC = np.asarray(XC)  # 模拟 邻近性的 数据 [?,6,32,32]
        XP = np.asarray(XP)  # 模拟 周期性的 数据 隔天
        XT = np.asarray(XT)  # 模拟 趋势性的 数据 隔周
        Y = np.asarray(Y)# [?,2,32,32]
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


if __name__ == '__main__':
    # depends = [range(1, 3 + 1),
    #            [1 * 48 * j for j in range(1, 3 + 1)],
    #            [7 * 48 * j for j in range(1, 3 + 1)]]
    # print(depends)
    # print([j for j in depends[0]])
    str = ['2013070101']
    t = string2timestamp(str)
    offset_frame = pd.DateOffset(minutes=24 * 60 // 48)  # 时间偏移 minutes = 30
    print(t)
    o = [t[0] - j * offset_frame for j in range(1, 4)]
    print(o)
