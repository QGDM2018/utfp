import pickle
import pandas as pd
import json
from datetime import timedelta
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

non = {'08:35',  '08:45', '15:40', '17:45', '13:40', '15:55', '10:50', '17:35',
       '12:50', '08:50', '13:55',  '13:35', '12:45', '12:55',
       '12:35',  '15:50', '13:50',  '17:50', '12:40',  '17:40',  '17:55',
       '08:40', '08:55', '13:45', '15:35', '10:45', '10:40',  '10:55', '15:45', '10:35'}


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
    1. get_roadflow_total： 某个路口，整个数据集的流量
    2. get_timeFlow： 某一天，所有路口的流量
    '''
    def __init__(self):
        try:
            with open('data/buffer.json', 'r', encoding='utf-8') as f:
                self.buffer = json.load(f)
        except:
            self.buffer = {}
            self.buffer['sCrossroadID'] = set(self.load_data(1)['crossroadID'])  # 第一天的roadid

    def __del__(self):
        # 保存缓存数据
        # with open('data/buffer.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.buffer, f)
        pass

    def load_data(self, i):
        '''从文件夹中载入csv表格'''
        encoding = 'utf-8'
        if i < 20:
            dirpath = 'data/trainCrossroadFlow/train_trafficFlow_'
        else:
            dirpath = 'data/testCrossroadFlow/test_trafficFlow_'
        return  pd.read_csv(dirpath+f'{i}.csv', encoding=encoding)[['timestamp', 'crossroadID', 'vehicleID']]

    def load_flow(self, i):
        '''载入并处理流量数据'''
        flow_df =self.load_data(i)
        flow_dct = self.cal_flow(flow_df)   # 计算流量
        with open(f'data/tmp/flow_{i}', 'wb') as f:
            pickle.dump(flow_dct, f)
        return flow_df, flow_dct

    def load_timeflow(self, i):
        flow_df = self.load_data(i)  # 'direction,laneID,timestamp,crossroadID,vehicleID'
        flow_dct = {}  # {road :[flow1,flow2] }
        flow_dfx = flow_df[['timestamp', 'crossroadID', 'vehicleID']]   # 时间，路口，车辆
        flow_dfx.loc[:, 'timestamp'] = pd.to_datetime(flow_dfx['timestamp'])  # str To TimeStamp
        flow_dfx['timestamp'] = flow_dfx['timestamp'].apply(round_minutes, n=5)  # 时间离散化，每五分钟
        for road, df in flow_dfx.groupby('timestamp'):  # 按路口分组
            df = df.drop(columns='timestamp')
            tgroup = df.groupby('crossroadID')
            flow_dct[road] = pd.Series([len(g[1]) for g in tgroup], index=tgroup.groups.keys())
        with open(f'data/tmp/timeFlow_{i}', 'wb') as f:
            pickle.dump(flow_dct, f)
        return flow_df, flow_dct

    def cal_flow(self, flow_dfx):
        '''计算流量'''
        flow_dct = {}  # {road :[flow1,flow2] }
        'direction,laneID,timestamp,crossroadID,vehicleID'
        flow_dfx = flow_dfx[['timestamp','crossroadID','vehicleID']]   # 时间，路口，车辆
        for road, df in flow_dfx.groupby('crossroadID'):      # 按路口分组
            df = df.drop(columns='crossroadID')   #
            df['timestamp'] = pd.to_datetime(df['timestamp'])   # str To TimeStamp
            df['timestamp'] = df['timestamp'].apply(round_minutes,n=5)  # 时间离散化，每五分钟
            tgroup = df.groupby('timestamp')
            flow_dct[road] = pd.Series([len(g[1]) for g in tgroup], index=tgroup.groups.keys())
        return flow_dct

    def get_roadflow(self, i) -> object:
        '''获取单个路口各个时段车流量数据
        :param i:天数
        :return:
            dfFlow：pd.DataFrame 原始车流数据表
            dFlow: {crossroadID:pd.Series} 车流量时序数据
        '''
        try:
            with open(f'data/tmp/flow_{i}','rb') as f:
                flow_dct = pickle.load(f)
            flow_df = self.load_data(i)
        except:
            flow_df, flow_dct = self.load_flow(i)
        return flow_df, flow_dct

    def get_roadflow_total(self, roadid):
        '''获取单个路口所有车流量数据
        :param roadId: 道路Id
        :return:
        '''
        try:
            with open(f'data/tmp/roadFlowTotal_{roadid}', 'rb') as f:
                return pickle.load(f)
        except:
            flow_df, flow_dct = self.get_roadflow(1)
            roadflow_df = flow_dct[roadid]  #
            sCrossroadID = set(flow_df['crossroadID'])
            for i in range(2, 24):   #
                flow_df, flow_dct = self.get_roadflow(i)
                roadflow_df = pd.concat([roadflow_df, flow_dct[roadid]])
                sCrossroadID.update(set(flow_df['crossroadID']))
            self.buffer['sCrossroadID'] = sCrossroadID
            # 缓存至文件
            with open(f'data/tmp/roadFlowTotal_{roadid}', 'wb') as f:
                pickle.dump(roadflow_df, f)
            return roadflow_df

    def get_timeFlow(self, i):
        '''获取某天单个时段各个路口车流量数据
        :param i:
        :return: dfFlow: 原始数据
                dFlow: { t : pd.Series([road1_flow, road2_flow])}
        '''
        try:
            with open(f'data/tmp/timeFlow_{i}', 'rb') as f:
                dFlow = pickle.load(f)
            dfFlow = self.load_data(i)
        except Exception as e:
            print(e)
            dfFlow,dFlow = self.load_timeflow(i)
        return dfFlow, dFlow


def dump_buffer():
    "统计单个路口所有车流量数据，并存入换成"
    prp = PreProcessor()
    freeze_support()
    from os import walk
    visited = set()
    for _, _, files in walk(r'E:\数据挖掘项目\qgTask\utfp\data\tmp'):
        for f in files:
            if 'road' in f:
                visited.add(f.split('_')[1])
    lRoadId = set(prp.load_data(1)['crossroadID'])
    lRoadId ^= visited
    pool = Pool(4)
    pool.map(prp.get_roadflow_total, lRoadId)
    # for r in lRoadId:
    #     print(r)
    #     prp.get_roadflow_total(r)
