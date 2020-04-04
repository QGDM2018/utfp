import numpy as np
import pandas as pd
from pre_process import get_adj_map, PreProcessor, freeze_support, Pool
import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity
from pre_process import get_trainroad_adjoin, get_testroad_adjoin


class FeatureEn:
    def __init__(self, term='first'):
        self.adj_map = get_adj_map()      # 对象与路网绑定
        self.prp = PreProcessor(term)  # 数据管理器

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
        X = np.zeros((d, len(lAdjNode))) # 相关系数矩阵,问题**每个路口训练一个模型
        return X

    def extract_adjoin_by_col(self):
        '''某时段内路口流量为特征，其邻接路口为目标值,构建训练集和测试集（按列遍历，很快）'''
        # 先提取训练集
        adj_map = {}  # {(flow, road): [{flow, roadID)}}
        data_dct = {'timestamp': [],
                    'crossroadID': [],
                    'mean_flow': [],
                    'flow': []}

        for road, adjoins in self.adj_map.items():
            adj_map[('flow', int(road))] = set(('flow', int(r)) for r in adjoins)
        flow_data = self.prp.load_buffer()
        if self.prp.term:  # 复赛情况
            data_dct['direction'] = []
            flow_data.set_index(['timestamp', 'direction', 'crossroadID'], inplace=True)
        else:
            flow_data.set_index(['timestamp', 'crossroadID'], inplace=True)
        flow_data = flow_data.unstack()
        flow_data.drop(columns=flow_data.columns ^ (flow_data.columns & adj_map.keys()), inplace=True)
        for key, values in adj_map.items():  # 缩小邻接表， 邻接卡口仅保留数据集出现的
            adj_map[key] = flow_data.columns & values
        for col in flow_data.columns:
            flow_data_nn = flow_data[flow_data[col].notna()]  # 去除空值
            if len(adj_map[col]):
                mean_flow = flow_data_nn[adj_map[col]].mean(axis=1).dropna()
                data_dct['crossroadID'].extend(col[1] for _ in range(len(mean_flow)))
                data_dct['mean_flow'].extend(mean_flow)
                data_dct['flow'].extend(flow_data_nn.loc[mean_flow.index, col])
                if self.prp.term:  # 复赛情况
                    for ts, dire in mean_flow.index:
                        data_dct['direction'].append(dire)
                        data_dct['timestamp'].append(ts)
                else:
                    data_dct['timestamp'].extend(mean_flow.index)
        # 获取测试集合
        if self.prp.term:
            test_df = self.prp.get_submit()[['crossroadID', 'direction', 'timeBegin', 'date']]
            test_df['timestamp'] = test_df[['timeBegin', 'date']].apply(
                lambda x: f'2019-{x["date"]} {x["timeBegin"].rjust(5, "0")}:00', axis=1)
            test_df.set_index(['timestamp', 'direction'], inplace=True)
        else:
            test_df = self.prp.get_submit()[['crossroadID', 'timeBegin', 'date']]
            test_df['timestamp'] = test_df[['timeBegin', 'date']].apply(
                lambda x: f'2019-08-{x["date"]} {x["timeBegin"].rjust(5, "0")}:00', axis=1)
            test_df.set_index('timestamp', inplace=True)
        return test_df, flow_data
        ts_index = test_df['timestamp'].unique()
        return ts_index, flow_data.index
        for road, indexes in test_df.groupby('crossroadID'):
            if ('flow', road) in adj_map:
                return ts_index, flow_data
                test_df.loc[indexes, 'mean_flow'] = flow_data.loc[ts_index, adj_map[('flow', road)]].mean(axis=1)
            else:
                test_df.loc[indexes, 'mean_flow'] = list(0 for _ in range(len(indexes)))
        # 邻接关系
        return pd.DataFrame(data_dct), test_df

    def similarity_matrix(self):
        '''
        计算卡口之间的相似度矩阵
        :return:相似度矩阵
        '''
        matrix, index = self.prp.get_roadflow_alltheday()
        cos = cosine_similarity(pd.DataFrame(np.array(matrix), index=index, columns=["1", "2", "3", "4", "5", "6", "7", "8"]))
        return cos, index

    def get_train_data(self):
        '''
        得到训练集和测试集
        :return:训练集和测试集
        '''
        global timelist
        train = self.prp.load_train()
        predMapping, mapping = get_testroad_adjoin(self.prp)
        train_mapping = get_trainroad_adjoin(predMapping, mapping)
        # [[邻居1[n天各个方向车流1]，邻居2，邻居3，……],[]]
        timelist = []
        for i in range(1, 22):  # 完整的时间列表
            timelist.extend(pd.date_range(f'2019/09/{i} 07:00', f'2019/09/{i} 18:55', freq='5min').tolist())
        # 修改train中的时间
        train["timestamp"] = [pd.to_datetime(i, errors='coerce') for i in train["timestamp"].tolist()]
        train["direction"] = [eval(i) for i in train["direction"].tolist()]
        # 整理数据
        train_x = []
        train_y = []
        for key in train_mapping.keys():
            a = []
            tdf = pd.DataFrame(timelist, columns=["timestamp"])     # 生成时间戳df
            tdf.to_csv("./data/tdf.csv")
            for i in train_mapping[key][:]:     # 相邻卡口
                # if train[train["crossroadID"] == i]["direction"].tolist():  # 若非空,存在a里面
                #     mean = np.array(train[train["crossroadID"] == i]["direction"].tolist()).mean(axis=0)
                #     mean = [int(round(x)) for x in mean[:]]
                #     input(mean)
                #     result = pd.merge(tdf, train[train["crossroadID"] == i], on='timestamp', how="left").drop("crossroadID", axis=1)
                #     result_ = []
                #     for y in result.fillna(str(mean))["direction"].tolist():
                #         if type(y) is str:
                #             y = eval(y)
                #         result_.append(y)
                #
                result_ = get_something(i, train, tdf)
                if result_:
                    a.append(result_)
            if a:   # 判断a中是否有内容
                train_x.append(a)   # 存入训练集[[[时间1],[时间2]],[],[]]
                train_y.append(get_something(key, train, tdf))   # 把key也加进来
        text_save("x", train_x)

    def get_text_data(self):
        train = self.prp.load_train()
        predMapping, mapping = get_testroad_adjoin(self.prp)
        test_x = []
        # [[邻居1[n天各个方向车流1]，邻居2，邻居3，……],[]]
        timelist = []
        keylst = []
        for i in range(1, 22):  # 完整的时间列表
            timelist.extend(pd.date_range(f'2019/09/{i} 07:00', f'2019/09/{i} 18:55', freq='5min').tolist())
        # 修改train中的时间
        train["timestamp"] = [pd.to_datetime(i, errors='coerce') for i in train["timestamp"].tolist()]
        train["direction"] = [eval(i) for i in train["direction"].tolist()]
        for key in predMapping.keys():
            keylst.append(key)
            a = []
            tdf = pd.DataFrame(timelist, columns=["timestamp"])  # 生成时间戳df
            for i in list(predMapping[key])[:]:  # 相邻卡口
                result_ = get_something(i, train, tdf)
                if result_:
                    a.append(result_)
                print(a)
            if a:   # 判断a中是否有内容
                test_x.append(a)   # 存入训练集[[[时间1],[时间2]],[],[]]
        text_save("test", test_x)
        return keylst


def text_save(flag, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    if flag == "x":
        filename = "./data/train_x.csv"
        s = []
        for i in range(len(data)):  # n个卡口
            for j in range(len(data[i][0])):   # 时间
                print(len(data[i][0]))
                a = []
                for k in range(len(data[i])):  # n个邻居
                    a.append(data[i][k][j])
                # a = str(a).replace("'", '').replace(',', '')  # 去除单引号，逗号，每行末尾追加换行符
                print(a)
                s.append(a)
        pd.DataFrame(s).to_csv(filename)
        print("train_x保存文件成功")
    elif flag == "y":
        filename = "./data/train_y.txt"
        f = open(filename, "a")
        s = []
        for i in range(len(data)):  # n个卡口
            for j in range(len(data[i])):   # 所有时段
                s.append(data[i][j])
        f.write(str(s))
        f.close()
        print("train_y保存文件成功")
    else:
        filename = "./data/test_x.csv"
        s = []
        for i in range(len(data)):  # n个卡口
            for j in range(len(data[i][0])):  # 时间
                print(len(data[i][0]))
                a = []
                for k in range(len(data[i])):  # n个邻居
                    a.append(data[i][k][j])
                # a = str(a).replace("'", '').replace(',', '')  # 去除单引号，逗号，每行末尾追加换行符
                print(a)
                s.append(a)
        pd.DataFrame(s).to_csv(filename)
        print("test保存文件成功")


def get_something(i, train, tdf):
    """  用均值填补缺失值，返回所有时间段的数据
    :param i: 某卡口
    :param train:训练集
    :param tdf:时间表
    :return: 各个时间段流量,result
    """
    b = []
    if train[train["crossroadID"] == i]["direction"].tolist():  # 若非空,存在a里面
        mean = np.array(train[train["crossroadID"] == i]["direction"].tolist()).mean(axis=0)
        mean = [int(round(x)) for x in mean[:]]
        result = pd.merge(tdf, train[train["crossroadID"] == i], on='timestamp', how="left").drop("crossroadID", axis=1)
        for y in result.fillna(str(mean))["direction"].tolist():
            if type(y) is str:
                y = eval(y)
            b.append(y)
    return b


