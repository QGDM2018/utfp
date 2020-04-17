import pandas as pd
import csv
import numpy as np
import datetime
from datetime import timedelta
from multiprocessing.pool import Pool
from multiprocessing import freeze_support
import tqdm

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
        self.flow_data, self.time_flow = None, None  # 流量数据
        self.train = None  # 训练数据
        self.train_x = None
        self.train_y = None
        self.test_x = None

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
            self.flow_data = pd.read_csv(f'./data/{self.term}_flow_data.csv')  # 若报错则先确认data目录结构，再调用dump_buffer
        return self.flow_data

    def load_train(self):
        if self.train is None:
            self.train = pd.read_csv(f'./data/train.csv', names=['crossroadID', 'timestamp', 'direction'])
        return self.train

    def cal_flow(self, i):
        '''计算流量'''
        flow_dfx = self.load_data(i)
        flow_dfx['timestamp'] = pd.to_datetime(flow_dfx['timestamp'])  # str To TimeStamp
        flow_dfx['timestamp'] = flow_dfx['timestamp'].apply(round_minutes, n=15)  # 时间离散化，每五分钟
        flow_lst = []  # [[road, timestamp, flow]]  [[road, timestamp, flow]]
        if not self.term:
            for road, df in flow_dfx.groupby('crossroadID'):  # 按路口分组
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
            # for i in path_dct['day_list'][self.term]:
            #     handler.writerows(self.cal_flow(i))

    def get_roadflow_alltheday(self):
        '''获取训练集中各个路口个个方向的车流量,用于预测相似度
        :return:matrix:卡口各个方向车流量
        '''
        flow_data = self.load_buffer()
        matrix, index = [], []
        a = [0, 0, 0, 0, 0, 0, 0, 0]
        for keys, df in flow_data.groupby(['crossroadID', 'direction']):
            if keys[0] in index:
                pass
            else:  # 出现新的卡口
                a = [0, 0, 0, 0, 0, 0, 0, 0]
            index.append(keys[0])
            a[int(keys[1]) - 1] = np.sum(df["flow"])
            matrix.append(a)
        df = pd.DataFrame({"index": index, "matrix": matrix}).drop_duplicates(subset=['index'], keep='last', inplace=False)
        return df['matrix'].tolist(), df['index'].tolist()

    def get_roadflow_by_road(self, roadid):
        '''获取单个路口所有车流量数据
        :param roadId: 道路Id
        :return:
        '''
        flow_data = self.load_buffer()
        flow_data = flow_data.set_index('timestamp')
        roadflow_df = flow_data[flow_data['crossroadID'] == roadid]
        if self.term:
            for dire, df in roadflow_df.groupby('direction'):
                yield dire, pd.Series(df['flow'])
        else:
            yield None, pd.Series(roadflow_df['flow'])

    def get_submit(self):
        submit = pd.read_csv(f'./data/submit/{self.term}_submit.csv')
        submit['timestamp'] = submit[['timeBegin', 'date']].apply(
            lambda x: f'2019-{x["date"]} {x["timeBegin"].rjust(5, "0")}:00', axis=1)
        return submit

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

    def fill_na(self):
        if self.term:
            flow_data = self.load_buffer()
            cur_day = datetime.datetime(2019, 9, 1, 7)
            unit_day = datetime.timedelta(days=1)
            five_minutes = datetime.timedelta(minutes=5)
            thirty_minutes = datetime.timedelta(minutes=30)
            train_ts = []
            for _ in range(21):
                cur_time = cur_day
                for _ in range(144):
                    train_ts.append(str(cur_time))
                    cur_time += five_minutes
                cur_day += unit_day
            test_ts = []
            for _ in range(4):
                cur_time = cur_day
                for i in range(1, 73):
                    test_ts.append(str(cur_time))
                    if i % 6 == 0:
                        cur_time += thirty_minutes
                    cur_time += five_minutes
                cur_day += unit_day
            ts_set_list = set(train_ts), set(test_ts)
            # 32
            flow_data_with_na = pd.DataFrame()
            for road, road_df in tqdm.tqdm(flow_data.groupby('crossroadID')):
                dire_lst = road_df['direction'].unique()
                data_list = []
                for ts_set in ts_set_list:
                    for ts in ts_set ^ (ts_set & set(road_df['timestamp'])):
                        # print(len(ts_set), len(set(road_df['timestamp'])), len(ts_set ^ (ts_set & set(road_df['timestamp']))))
                        # return ts_set,  set(road_df['timestamp'])
                        for dire in dire_lst:
                            data_list.append([road, dire, ts, 0])
                            # flow_data.loc[cur_index] = [road, dire, ts, 0]
                flow_data_with_na = pd.concat(
                    (flow_data_with_na, pd.DataFrame(data_list, columns=['crossroadID', 'direction', 'timestamp', 'flow']))
                    , axis=0, ignore_index=True)
            flow_data_with_na = pd.concat((flow_data_with_na, flow_data), axis=0, ignore_index=True)
            flow_data_with_na.to_csv("./data/flow_data_with_na.csv", index=False)  # 3783710; 4814894
            return flow_data_with_na
        # b = a[a.crossroadID == 100002]
        # b = b[b.direction == 1].timestamp.values
        # b.sort()
        # print(b[-200:])

    # 获取训练集的样子
    def get_train_data(self):
        flow_data = self.load_buffer()
        # ['crossroadID', 'timestamp',[八个方向]]
        flow_list = []
        for keys, df in flow_data.groupby(['crossroadID', 'timestamp']):  # 按卡口、时间分组
            a = [0, 0, 0, 0, 0, 0, 0, 0]
            for index, row in df.iterrows():
                a[row[1] - 1] = row[-1]
            flow_list.append([*keys, a])
        pd.DataFrame(flow_list).to_csv("data/train.csv", encoding="utf-8")

    def load_traindata(self):
        self.train_x = pd.read_csv("./data/train_x.csv")
        self.train_y = open("./data/train_y.txt").read()
        self.test_x = pd.read_csv("./data/test_x.csv")
        self.train_y = eval(self.train_y)  # list类型
        self.train_x = self.changetype(self.train_x)
        self.test_x = self.changetype(self.test_x)
        return self.train_x, self.train_y, self.test_x

    def changetype(self, data):
        """
        将dataframe中的类型改变str->list,并不上缺失值
        :param data:
        :return:
        """
        total = []
        for row in data.fillna(str([0, 0, 0, 0, 0, 0, 0, 0])).iterrows():
            a = []
            for i in range(1, len(row[1])):
                a.append(eval(row[1][i]))
            total.append(a)
        return pd.DataFrame(total)


# 获取测试卡口的相邻卡口
def get_testroad_adjoin(prp):
    # 邻接表,无去重
    mapping = get_adj_map()
    # 获取第一个邻接节点
    sPredRoad = set(prp.get_submit()['crossroadID'])  # 要预测的路口
    predMapping = {}
    # available 只在训练集出现的邻接卡口
    available = set(prp.load_buffer()['crossroadID']) ^ (sPredRoad & set(prp.load_buffer()['crossroadID']))
    for r in sPredRoad:  # 要预测的每一个卡口
        vs = mapping.get(r)  # 邻接表中每一个键（卡口），返回键的值-->即邻接卡口
        if vs is not None:
            adj_set = set(vs)
            bind = adj_set & available
            if bind:
                predMapping[r] = bind.pop()  # 保留在训练集中出现过的卡口
    rest = sPredRoad ^ predMapping.keys()  # 不出现在训练集中的卡口，随机找相邻
    for r in rest:      # 空值处理
        predMapping[r] = None
    return predMapping
    # return rest, predMapping
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


def get_trainroad_adjoin(premap, map):
    train_id = pd.read_csv(f'./data/train.csv', names=['crossroadID', 'timestamp', 'direction'])["crossroadID"].tolist()
    train_map = {}
    for key in map.keys():
        if key in train_id:
            train_map[key] = list(set(map[key]))
    train_mapping = train_map.copy()
    for key in train_map.keys():
        if [x for x in train_map[key] if x in list(premap.keys())]:  # 邻接值需要被预测
            try:
                train_mapping.pop(key)
            except IndexError as e:
                continue
    return train_mapping  # 得到训练集的卡口


def get_adj_map():
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
    return adj_map


def get_testroad_adjoin_lr(prp):
    # 邻接表,无去重
    mapping = get_adj_map()
    # 获取第一个邻接节点
    sPredRoad = set(prp.get_submit()['crossroadID'])  # 要预测的路口
    predMapping = {}
    # available 只在训练集出现的邻接卡口
    available = {100097, 100354, 100355, 100227, 100359, 100360, 100105, 100237, 100117, 100118, 100375, 100377, 100378, 100252,
    100381, 100382, 100388, 100389, 100134, 100007, 100264, 100137, 100145, 100222, 100152, 100153, 100283, 100284,
    100157, 100030, 100031, 100158, 100160, 100161, 100291, 100036, 100295, 100045, 100303, 100176, 100306, 100051,
    100052, 100181, 100056, 100057, 100058, 100319, 100578, 100452, 100453, 100326, 100327, 100331, 100332, 100077,
    100205, 100208, 100209, 100211, 100213, 100472, 100094}
    for r in sPredRoad:  # 要预测的每一个卡口
        queue = list(mapping.get(r, [])).copy()
        while queue:
            cur = queue.pop(0)
            if cur in available:
                predMapping[r] = cur  # 保留在训练集中出现过的卡口
                break
            else:
                queue.extend(mapping.get(cur, []))
    rest = sPredRoad ^ predMapping.keys()  # 不出现在训练集中的卡口，随机找相邻
    for r in rest:      # 空值处理
        predMapping[r] = None
    return predMapping
