import numpy as np
import pandas as pd
from pre_process.pre_process import get_adj_map, PreProcessor
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from pre_process.pre_process import get_trainroad_adjoin, get_testroad_adjoin


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
        if self.prp.term:  # 复赛情况
            # 加载数据集
            flow_data = self.prp.load_buffer()  # 训练集
            road_set = set(flow_data['crossroadID'])
            # 构建邻接表
            road_direction_dct = {}  # 用于构建邻接表
            for road in road_set:
                road_direction_dct[road] = flow_data['direction'][flow_data['crossroadID'] == road].unique()
            adj_map = {}  # {road: {'adjoin': [('flow', 'roadID', 'direction')], 'self':[]}}
            for road in road_set:
                adjoin_set = set(self.adj_map[road]) & road_set
                adj_map[road] = set()
                for adjoin in adjoin_set:
                    adj_map[road].update(('flow', dire, road) for dire in road_direction_dct[adjoin])
            flow_data.set_index(['timestamp', 'crossroadID', 'direction'], inplace=True)
            flow_data = flow_data.unstack().unstack()  # 重建列的索引
            # flow_data.drop(columns=flow_data.columns ^ (flow_data.columns & adj_map.keys()), inplace=True)
            # 获取训练集
            train_index = flow_data.index < '2019-09-22 07:00:00'
            train_flow = flow_data[train_index]  # 划分数据集, 训练集
            train_x_index = train_flow.index[train_flow.index < '2019-09-21 18:30:00']  # 训练集特征索引
            train_y_index = (pd.to_datetime(train_x_index) + datetime.timedelta(minutes=30)
                             ).map(lambda x: str(x)) & flow_data.index
            train_x_index &= (pd.to_datetime(train_y_index) - datetime.timedelta(minutes=30)
                              ).map(lambda x: str(x))  # 训练集目标值索引
            train_flow_x = train_flow.loc[train_x_index]
            train_flow_y = train_flow.loc[train_y_index]
            # 获取测试集索引
            test_flow = flow_data[~train_index]
            submit_data = self.prp.get_submit()
            test_index_y = submit_data['timestamp'].unique()
            test_index_x = (pd.to_datetime(test_index_y) - datetime.timedelta(minutes=30)
                            ).map(lambda x: str(x)) & flow_data.index  # 测试索引
            test_flow = test_flow.loc[test_index_x]  # 测试集
            for road in road_set:
                adjoin_cols = adj_map[road]
                if len(adjoin_cols):
                    # 根据邻接表提取 训练集(X, direction, flow)
                    train_df = pd.DataFrame()
                    x_cloumns = list(i[1:] for i in adjoin_cols)  # 新的单索引列名，防止报错
                    for dire in road_direction_dct[road]:  # 先纵向扩充df
                        train_df_next = train_flow_x[adjoin_cols]
                        train_df_next.columns = x_cloumns
                        train_df_next['direction'] = [dire] * len(train_df_next)
                        train_df_next['y'] = train_flow_y[('flow', dire, road)].values
                        train_df = pd.concat((train_df, train_df_next[train_df_next['y'].notna()]), axis=0)
                    train_df = pd.concat((train_df, pd.get_dummies(train_df['direction'])), axis=1)  # 再将每个方向作为一列（哑编码）
                    # 根据邻接表提取  (X, direction, flow)
                    test_df = pd.DataFrame()
                    for dire in road_direction_dct[road]:  # 先纵向扩充df
                        test_df_next = test_flow[adjoin_cols]
                        test_df_next.columns = x_cloumns
                        test_df_next.index = test_index_y
                        test_df_next['direction'] = [dire] * len(test_df_next)
                        test_df = pd.concat((test_df, test_df_next), axis=0)
                    test_df = pd.concat((test_df, pd.get_dummies(test_df['direction'])), axis=1)  # 再将每个方向作为一列（哑编码）
                    # 去除空的列
                    for df in (train_df, test_df):
                        na_index = df.isna().sum(axis=0)
                        for col in na_index[na_index == len(df)].index:  #
                            train_df.drop(columns=col, inplace=True)
                            test_df.drop(columns=col, inplace=True)
                    yield road, train_df, test_df

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
