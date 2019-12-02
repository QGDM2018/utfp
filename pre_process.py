from datetime import timedelta
import pickle
# 时间离散化，向下取整
def tZoneMode(t,n):
    '''时间区间，向下取整
    :param t: 时间戳
    :param n: 取余的数
    :return:
    '''
    # 对分钟取整
    return t - timedelta(minutes=t.minute%n,seconds=t.second)

import pandas as pd
class PreProcessor:
    def __init__(self):
        # self.trainFlow =  self.load_flow(1)
        pass
    def load_csv(self,i):
        '''从文件夹中载入csv表格'''
        encoding = 'utf-8'
        if i < 20:
            dirpath = 'data/trainCrossroadFlow/train_trafficFlow_'
        else:
            dirpath = 'data/testCrossroadFlow/test_trafficFlow_'
        return  pd.read_csv(dirpath+f'{i}.csv',encoding=encoding)   # 读取1号的

    def load_flow(self,i):
        '''载入并处理流量数据'''
        dfFlow =self.load_csv(i)
        dFlow =  self.cal_flow(dfFlow)   # 计算流量
        with open(f'data/tmp/flow_{i}','wb') as f:
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
            with open('data/tmp/flow_'+i,'wb') as f:
                dFlow = pickle.load(f)
            dfFlow = self.load_csv(i)
        except:
            dfFlow,dFlow = self.load_flow(i)
        return dfFlow,dFlow

    def get_timeFlow(self,i):
        '''获取单个时段各个路口车流量数据
        :param i:
        :return:
        '''
        return