import pandas as pd
import csv
import json
from datetime import timedelta
from multiprocessing.pool import Pool
from multiprocessing import freeze_support


"20、21、23天，卡口100306缺失"
# 目录：训练集、测试集、缓存数据；如何设置根目录？

path_dct = {'trainflow': ['./data/first/trainCrossroadFlow/train_trafficFlow_%d',
                          './data/final/train/train_trafficFlow_09-%02d'],
            'testflow': ['./data/first/testCrossroadFlow/test_trafficFlow_%d',
                         './data/final/test_user/test_trafficFlow_09-%02d'],
            'columns': [['crossroadID', 'timestamp', 'flow'], ['crossroadID', 'direction', 'timestamp', 'flow']],
            'day_list': [list(range(3, 24)) + [1], range(1, 26)]}


def round_minutes(t, n):
    '''时间区间，向下取整
    :param t: 时间戳
    :param n: 取余的数
    :return:
    '''
    # 对分钟取整
    return t - timedelta(minutes=t.minute % n, seconds=t.second)


class PreProcessor:
    '''预处理：从原数据中统计每五分钟，各个时段、各个路口的流量
        dump_buffer :统计卡口流量
    '''
    def __init__(self, term='first'):
        self.term = 0 if term == 'first' else 1
        self.flow_path_lst = []
        try:
            with open(f'./data/{self.term}_buffer.json', 'r', encoding='utf-8') as f:
                self.buffer = json.load(f)
        except:
            self.buffer = {'sCrossroadID': list(set(self.load_data(3)['crossroadID']))}  # 第一天的roadid
        self.flow_data = None  # 流量数据
    def __del__(self):
        # 保存缓存数据
        with open(f'./data/{self.term}_buffer.json', 'w', encoding='utf-8') as f:
            json.dump(self.buffer, f)

    def load_data(self, i):
        '''从文件夹中载入csv表格'''
        encoding = 'utf-8'
        border = 22 if self.term else 20
        if i < border:
            dirpath = path_dct['trainflow'][self.term]
        else:
            dirpath = path_dct['testflow'][self.term]
        columns = ['timestamp', 'crossroadID', 'vehicleID']
        if self.term:
            columns.append('direction')
        return pd.read_csv(dirpath % i + '.csv', encoding=encoding)

    def load_buffer(self):
        if self.flow_data is None:
            self.flow_data = pd.read_csv(f'./data/{self.term}_flow_data.csv')  # 若保存则需调用dump_buffer
        return self.flow_data

    def cal_flow(self, i):
        '''计算流量'''
        flow_dfx = self.load_data(i)
        flow_dfx['timestamp'] = pd.to_datetime(flow_dfx['timestamp'])  # str To TimeStamp
        flow_dfx['timestamp'] = flow_dfx['timestamp'].apply(round_minutes, n=5)  # 时间离散化，每五分钟
        # flow_dct = {}  # {road :[flow1,flow2] }
        flow_lst = []  # [[road, timestamp, flow]]  [[road, timestamp, flow]]
        if not self.term:
            for road, df in flow_dfx.groupby('crossroadID'):      # 按路口分组
                flow_lst.extend([road, g[0], len(g[1])] for g in df.groupby('timestamp'))
            return flow_lst
        else:
            for keys, df in flow_dfx.groupby(['crossroadID', 'direction']):  # 按路口分组
                flow_lst.extend([*keys, g[0], len(g[1])] for g in df.groupby('timestamp'))
            return flow_lst

    def dump_buffer(self, num=4):
        '''统计卡口流量, 存入csv， columns = ['crossroadID', 'timestamp', 'flow']
        :param num: 进程的数量
        :return:
        '''
        freeze_support()
        pool = Pool(num)
        with open(f'./data/{self.term}_flow_data.csv', 'w', newline='') as f:
            handler = csv.writer(f)
            handler.writerow(path_dct['columns'][self.term])
            for flow_lst in pool.map(self.cal_flow, path_dct['day_list'][self.term]):
                handler.writerows(flow_lst)

    def get_timeflow_by_day(self, i):
        '''获取某天单个时段各个路口车流量数据
        :param i:
        :return: dfFlow: 原始数据
                dFlow: { t : pd.Series([road1_flow, road2_flow])}
        '''
        flow_dct = {}  #
        return flow_dct

    def get_roadflow_by_day(self, i):
        '''获取单个路口各个时段车流量数据
        :param i:天数
        :return: dFlow: {crossroadID:pd.Series} 车流量时序数据
        '''
        flow_dct = {}  # {crossroadID: pd.Series}
        return flow_dct

    def get_roadflow_by_road(self, roadid):
        '''获取单个路口所有车流量数据
        :param roadId: 道路Id
        :return:
        '''
        flow_data = self.load_buffer()
        roadflow_df = flow_data[flow_data['crossroadID'] == roadid]
        return pd.Series(roadflow_df['flow'].values, index=roadflow_df['timestamp'])


    def roadid_nums(self):
        '''查看各天的记录roadid数量'''
        if self.term:
            day_list = list(range(1, 26))
        else:
            day_list = list(range(3, 24))
        data = []
        for d in day_list:
            ids = set(self.load_data(d)['crossroadID'])
            data.append((len(ids), ids))
        return data


def get_testroad_adjoin(prp):
    first = [100115, 100245, 100246, 100374, 100249, 100003, 100004, 100397, 100019, 100020, 100285, 100159, 100287,
             100288, 100164, 100041, 100300, 100434, 100179, 100180, 100053, 100183, 100315, 100316, 100061, 100193,
             100066, 100451, 100069, 100200, 100329, 100457, 100340, 100343, 100217]
    # 邻接表
    mapping = {}
    df = pd.read_csv('data/first/trainCrossroadFlow/roadnet.csv')
    for arr in df.values:
        h, t = arr[0], arr[1]
        mapping[h] = mapping.get(h, []) + [t]
        mapping[t] = mapping.get(t, []) + [h]
    # 获取第一个邻接节点
    sPredRoad = set(pd.read_csv('data/first/testCrossroadFlow/submit_example.csv')['crossroadID'])  # 要预测的路口
    predMapping = {}
    available = set(prp.buffer['sCrossroadID']) ^ (sPredRoad & set(prp.buffer['sCrossroadID']))  # 邻接节点的卡口为测试集中的卡口
    for r in sPredRoad:
        vs = mapping.get(r)
        if vs is not None:
            adj_set = set(vs)
            bind = adj_set & available
            if bind:
                predMapping[r] = bind.pop()
    rest = sPredRoad ^ predMapping.keys()
    # 训练集的数据
    length = len(rest)
    while True:
        for r in rest:
            adi = set(mapping.get(r, [])) & predMapping.keys()
            if adi:
                predMapping[r] = predMapping[adi.pop()]  # 暂时是用待预测路口的数据
        rest = sPredRoad ^ predMapping.keys()
        if length == len(rest):
            break  # 如果没有变化则跳出
        length = len(rest)
    candi = list(predMapping.values())[0]
    for roadid in rest:
        predMapping[roadid] = candi
    return predMapping
