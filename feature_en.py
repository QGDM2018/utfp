import numpy as np
# def s_t_relevancy(dFlow,i,j,d):
#     ''''''


class FeatureEn:
    def __init__(self, adjMap):
        self.adjMap = adjMap      # 对象与路网绑定

    def extract_relevancy(self, roadId, d, dFlow):
        '''抽取车流量表相关度，作为训练集
        :param roadId: 路口
        :param d: 时间延迟
        :param dFlow: 车流量表，{roadId:pd.Series}
        :param adjMap:
        :return: X，相关度矩阵，每一列代表一个样本，即横向空间维度，纵向时间维度
        '''

        # 时空相关系数计算
        lAdjNode = self.adjMap[roadId]
        X = np.zeros((d,len(lAdjNode))) # 相关系数矩阵,,问题**每个路口训练一个模型
        return