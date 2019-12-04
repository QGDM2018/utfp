from datetime import timedelta
import pickle
# from tqdm import tqdm
# 时间离散化，向下取整
def tZoneMode(t,n):
    '''时间区间，向下取整
    :param t: 时间戳
    :param n: 取余的数
    :return:
    '''
    # 对分钟取整
    return t - timedelta(minutes=t.minute%n,seconds=t.second)

def saving():
    first = [100115, 100245, 100246, 100374, 100249, 100003, 100004, 100397, 100019, 100020, 100285, 100159, 100287,
             100288, 100164, 100041, 100300, 100434, 100179, 100180, 100053, 100183, 100315, 100316, 100061, 100193,
             100066, 100451, 100069, 100200, 100329, 100457, 100340, 100343, 100217]
    mapping = {}
    sPredRoad = set(pd.read_csv('data/testCrossroadFlow/submit_example.csv')['crossroadID'])
    df = pd.read_csv('data/trainCrossroadFlow/roadnet.csv')
    for arr in df.values:
        h, t =arr[0],arr[1]
        mapping[h] = mapping.get(h,[]) + [t]
    predMapping = {}
    for r in sPredRoad:
        vs = mapping.get(r)
        if vs is not None:
            predMapping[r] = vs[0]
        # else:
        #     print(r)
    return predMapping

import pandas as pd
import json
class PreProcessor:
    def __init__(self):
        self.init_sys()

    def init_sys(self):
        # 加载缓存数据
        try:
            with open('data/buffer.json','r',encoding='utf-8') as f:
                self.buffer = json.load(f)
        except:
            self.buffer = {}
            self.buffer['sCrossroadID'] = set(self.load_csv(1)['crossroadID'])

        # self.lDfFlow = []
        # for i in tqdm(range(1, 24)):  #
        #     self.lDfFlow.append(self.load_csv(i))


    def __del__(self):
        # 保存缓存数据
        with open('data/buffer.json', 'w', encoding='utf-8') as f:
            json.dump(self.buffer,f)

    def load_csv(self,i):
        '''从文件夹中载入csv表格'''
        encoding = 'utf-8'
        if i < 20:
            dirpath = 'data/trainCrossroadFlow/train_trafficFlow_'
        else:
            dirpath = 'data/testCrossroadFlow/test_trafficFlow_'
        return  pd.read_csv(dirpath+f'{i}.csv',encoding=encoding)[['timestamp','crossroadID','vehicleID'] ]  # 读取1号的

    def load_flow(self,i):
        '''载入并处理流量数据'''
        dfFlow =self.load_csv(i)
        dFlow =  self.cal_flow(dfFlow)   # 计算流量
        with open(f'data/tmp/flow_{i}','wb') as f:
            pickle.dump(dFlow,f)
        return dfFlow,dFlow
    def load_timeFlow(self,i):
        dfFlow =self.load_csv(i)
        dFlow = {}  # {road :[flow1,flow2] }
        'direction,laneID,timestamp,crossroadID,vehicleID'
        dfFlowx = dfFlow[['timestamp','crossroadID','vehicleID']]   # 时间，路口，车辆
        dfFlowx.loc[:,'timestamp'] = pd.to_datetime(dfFlowx['timestamp'])  # str To TimeStamp
        dfFlowx['timestamp'] = dfFlowx['timestamp'].apply(tZoneMode, n=5)  # 时间离散化，每五分钟
        for road,df in dfFlowx.groupby('timestamp'):      # 按路口分组
            df = df.drop(columns = 'timestamp')   #
            tgroup = df.groupby('crossroadID')
            lFlow = [len(g[1]) for g in tgroup]  # 流量序列
            dFlow[road] = pd.Series(lFlow,index=tgroup.groups.keys())
        with open(f'data/tmp/timeFlow_{i}','wb') as f:
            pickle.dump(dFlow,f)
        return dfFlow,dFlow


    def cal_flow(self,dfFlow):
        '''计算流量'''
        dFlow = {}  # {road :[flow1,flow2] }
        'direction,laneID,timestamp,crossroadID,vehicleID'
        dfFlowx = dfFlow[['timestamp','crossroadID','vehicleID']]   # 时间，路口，车辆
        for road,df in dfFlowx.groupby('crossroadID'):      # 按路口分组
            df = df.drop(columns = 'crossroadID')   #
            df['timestamp'] = pd.to_datetime(df['timestamp'])   # str To TimeStamp
            df['timestamp'] = df['timestamp'].apply(tZoneMode,n=5)  # 时间离散化，每五分钟
            tgroup = df.groupby('timestamp')
            lFlow = [len(g[1]) for g in tgroup]  # 流量序列
            dFlow[road] = pd.Series(lFlow,index=tgroup.groups.keys())
        return dFlow

    def get_roadFlow(self,i) -> object:
        '''获取单个路口各个时段车流量数据
        :param i:天数
        :return:
            dfFlow：pd.DataFrame 原始车流数据表
            dFlow: {crossroadID:pd.Series} 车流量时序数据
        '''
        try:
            with open(f'data/tmp/flow_{i}','rb') as f:
                dFlow = pickle.load(f)
            dfFlow = self.load_csv(i)
        except:
            dfFlow,dFlow = self.load_flow(i)
        return dfFlow,dFlow


    def get_roadFlow_total(self,roadId):
        '''获取单个路口所有车流量数据
        :param roadId: 道路Id
        :return:
        '''
        try:
            with open(f'data/tmp/roadFlowTotal_{roadId}','rb') as f:
                return pickle.load(f)
        except:
            dfFlow, dFlow =  self.get_roadFlow(1)
            roadFlow = dFlow[roadId]  #
            sCrossroadID = set(dfFlow['crossroadID'])
            for i in range(2,24):   #
                dfFlow, dFlow = self.get_roadFlow(i)
                roadFlow = pd.concat([roadFlow,dFlow[roadId]])
                sCrossroadID.update(set(dfFlow['crossroadID']))
            self.buffer['sCrossroadID'] = sCrossroadID
            # 缓存至文件
            with open(f'data/tmp/roadFlowTotal_{roadId}','wb') as f:
                pickle.dump(roadFlow,f)
            return roadFlow

    def get_timeFlow(self,i):
        '''获取单个时段各个路口车流量数据
        :param i:
        :return:
        '''
        try:
            with open(f'data/tmp/timeFlow_{i}','rb') as f:
                dFlow = pickle.load(f)
            dfFlow = self.load_csv(i)
        except Exception as e:
            print(e)
            dfFlow,dFlow = self.load_timeFlow(i)
        return dfFlow,dFlow

    # 待测
    def loadm_flow(self,i):
        '''载入并处理流量数据'''
        dfFlow =self.lDfFlow[i]
        dFlow =  self.cal_flow(dfFlow)   # 计算流量
        with open(f'data/tmp/flow_{i}','wb') as f:
            pickle.dump(dFlow,f)
        return dfFlow,dFlow
    def getm_roadFlow(self,i) -> object:
        '''获取单个路口各个时段车流量数据
        :param i:天数
        :return:
            dfFlow：pd.DataFrame 原始车流数据表
            dFlow: {crossroadID:pd.Series} 车流量时序数据
        '''
        try:
            with open(f'data/tmp/flow_{i}','rb') as f:
                dFlow = pickle.load(f)
            dfFlow = self.lDfFlow[i]
        except:
            dfFlow,dFlow = self.load_flow(i)
        return dfFlow,dFlow