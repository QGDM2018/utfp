import numpy as np
import pandas as pd
# def s_t_relevancy(dFlow,i,j,d):
#     ''''''
import tqdm


class FeatureEn:
    def __init__(self, prp):
        adj_map = {}
        net_df = pd.read_csv('data/first/trainCrossroadFlow/roadnet.csv')
        for h, t in net_df.values:
            if h in adj_map:
                adj_map[h].add(t)
            else:
                adj_map[h] = {t}
            if t in adj_map:
                adj_map[t].add(h)
            else:
                adj_map[t] = {h}
        self.adj_map = adj_map      # 对象与路网绑定
        self.prp = prp  # 数据管理器

    def extract_relevancy(self, roadId, d, dFlow):
        '''抽取车流量表相关度，作为训练集
        :param roadId: 路口
        :param d: 时间延迟
        :param dFlow: 车流量表，{roadId:pd.Series}
        :param adjMap:
        :return: X，相关度矩阵，每一列代表一个样本，即横向空间维度，纵向时间维度
        '''

        # 时空相关系数计算
        lAdjNode = self.adj_map[roadId]
        X = np.zeros((d,len(lAdjNode))) # 相关系数矩阵,,问题**每个路口训练一个模型
        return

    def extract_adjoin(self):
        '''某时段内路口流量为特征，其邻接路口为目标值'''
        data = {}
        for road in tqdm.tqdm(self.prp.buffer['sCrossroadID']):
            data[road] = self.prp.get_roadflow_by_road(road)
        columns = ['x', 'y']
        x, y = [], []
        for node, adj_lst in self.adj_map.items():
            node_series = data[node]
            for adjoin in adj_lst:
                adj_series = data[adjoin]
                same = node_series.index & adj_series.index
                x.extend(node_series)
        return data
