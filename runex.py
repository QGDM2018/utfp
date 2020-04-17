from pre_process.pre_process import PreProcessor, get_testroad_adjoin, pd, get_testroad_adjoin_lr
import matplotlib.pyplot as plt
from model.ARMA import predict
from feature_en.feature_en import FeatureEn
import tqdm
from model.AP import ap_predict

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


def arma_ex(term='first'):
    prp = PreProcessor(term)  # 数据管理器
    preMapping = get_testroad_adjoin(prp)
    submit_df = prp.get_submit()
    dire_dct = dict([road, list(set(df['direction']))] for (road, df) in submit_df.groupby('crossroadID'))
    submit_index, day_list = [], range(22, 26)  # 索引
    for day in day_list:
        submit_index.extend(pd.date_range(start=f'2019-09-{day} 07:00:00', periods=144, freq='5min'))
    predict_df = pd.DataFrame()
    for pre_id in tqdm.tqdm(preMapping.keys()):
        instand_id = preMapping[pre_id]
        dire_list = dire_dct[pre_id].copy()
        for dire, roadflow in prp.get_roadflow_by_road(instand_id):
            if not dire_list:
                break  # 如空了则不要了，
            try:
                pred_pre = predict(roadflow)
            except Exception as e:
                print(instand_id, '\t', e)
                pred_pre = pd.Series([0.1] * (144 * 4), index=submit_index)
            # pred_pre = pred_pre.dropna(axis=0, how="any")
            pred_pre.fillna(pred_pre.mean(), inplace=True)
            for i in range(len(pred_pre)):
                pred_pre.iloc[[i]] = int(pred_pre.iloc[[i]])
            pred = pd.DataFrame(pred_pre.values, columns=['value'])
            pred['timestamp'] = submit_index
            pred['date'] = pred['timestamp'].apply(lambda x: x.strftime('%d'))
            pred['timeBegin'] = pred['timestamp'].apply(lambda x: f'{x.hour}:{x.strftime("%M")}')
            pred['crossroadID'] = pre_id
            pred['min_time'] = pred['timestamp'].apply(lambda x: int(x.strftime('%M')))
            pred = pred[pred['min_time'] >= 30]
            pred.drop(['timestamp'], axis=1, inplace=True)
            order = ['date', 'crossroadID', 'timeBegin', 'value']
            pred = pred[order]
            if prp.term:
                pred['direction'] = dire_list.pop()
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
        while dire_list:  # 方向不够用的情况
            pred['direction'] = dire_list.pop()
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
    submit_time_set = set(submit_df['timeBegin'])
    predict_df.set_index('timeBegin', inplace=True)
    return predict_df.loc[list(set(predict_df.index) & submit_time_set)].reset_index()[
        ['date', 'crossroadID', 'direction', 'timeBegin', 'value']]


def ap(f):
    """
    聚类算法
    :param f:FeatureEn
    :return:训练集中各个卡口的类别(字典类型)，
    :center_id:中心点信息
    """
    cos, index = f.similarity_matrix()
    cluster_centers_indices, labels = ap_predict(cos)
    id_type = dict(zip(index, labels))
    center_id = dict(zip([i+1 for i in range(len(cluster_centers_indices))], cluster_centers_indices))
    print(center_id, id_type)
    return id_type, center_id


def regression_ex(term='final'):
    keylst = [
        100115, 100245, 100246, 100374, 100003, 100004, 100020, 100285, 100159, 100287, 100288, 100164, 100300, 100179,
        100053, 100183, 100315, 100061, 100193, 100066, 100457, 100343, 100217, 100434, 100249, 100316, 100329, 100019,
        100340, 100041, 100069
    ]
    keylst = [val for val in keylst for i in range(3024)]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    prp = PreProcessor(term)  # 数据管理器
    train_x, train_y, test_x = prp.load_traindata()
    # 训练模型
    lr = LinearRegression()
    # print(train_x.iloc[:, 0:1])
    lr.fit(train_x.iloc[:, 0:1].values, train_y)
    test_y = lr.predict(test_x.values)
    # print(test_y)
    return test_y


def regression_ex_vvlj(term='first'):
    prp = PreProcessor(term)  # 数据管理器
    preMapping = get_testroad_adjoin(prp)
    submit_df = prp.get_submit()
    dire_dct = dict([road, list(set(df['direction']))] for (road, df) in submit_df.groupby('crossroadID'))
    submit_index, day_list = [], range(22, 26)  # 索引
    for day in day_list:
        submit_index.extend(pd.date_range(start=f'2019-09-{day} 07:00:00', periods=144, freq='5min'))
    predict_df = pd.DataFrame()
    for pre_id in tqdm.tqdm(preMapping.keys()):
        instand_id = preMapping[pre_id]
        dire_list = dire_dct[pre_id].copy()
        for dire, roadflow in prp.get_roadflow_by_road(instand_id):
            if not dire_list:
                break  # 如空了则不要了，
            try:
                pred_pre = predict(roadflow)
            except Exception as e:
                print(instand_id, '\t', e)
                pred_pre = pd.Series([0.1] * (144 * 4), index=submit_index)
            # pred_pre = pred_pre.dropna(axis=0, how="any")
            pred_pre.fillna(pred_pre.mean(), inplace=True)
            for i in range(len(pred_pre)):
                pred_pre.iloc[[i]] = int(pred_pre.iloc[[i]])
            pred = pd.DataFrame(pred_pre.values, columns=['value'])
            pred['timestamp'] = submit_index
            pred['date'] = pred['timestamp'].apply(lambda x: x.strftime('%d'))
            pred['timeBegin'] = pred['timestamp'].apply(lambda x: f'{x.hour}:{x.strftime("%M")}')
            pred['crossroadID'] = pre_id
            pred['min_time'] = pred['timestamp'].apply(lambda x: int(x.strftime('%M')))
            pred = pred[pred['min_time'] >= 30]
            pred.drop(['timestamp'], axis=1, inplace=True)
            order = ['date', 'crossroadID', 'timeBegin', 'value']
            pred = pred[order]
            if prp.term:
                pred['direction'] = dire_list.pop()
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
        while dire_list:  # 方向不够用的情况
            pred['direction'] = dire_list.pop()
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
    submit_time_set = set(submit_df['timeBegin'])
    predict_df.set_index('timeBegin', inplace=True)
    return predict_df.loc[list(set(predict_df.index) & submit_time_set)].reset_index()[
        ['date', 'crossroadID', 'direction', 'timeBegin', 'value']]


def regression_many_x(term='final'):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    prp = PreProcessor(term)
    pred_map = get_testroad_adjoin_lr(prp)
    submit_df = prp.get_submit()
    submit_df.set_index(['timestamp', 'direction', 'crossroadID'], inplace=True)
    fe = FeatureEn(term)
    r2_rst = {}
    predict_dct = {}
    for road, train_data, test_data in fe.extract_adjoin_by_col():
        train_data = train_data.dropna(axis=0)
        # print(test_data.isna().sum(0))
        test_data = test_data.dropna(axis=0)
        X, y = train_data.drop(columns=['y', 'direction']), train_data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        rst = r2_score(y_test, lr.predict(X_test))
        r2_rst[road] = rst
        test_data['flow'] = lr.predict(test_data.drop(columns='direction'))
        # 格式化
        test_data = test_data.reset_index()
        test_data['crossroadID'] = road
        predict_dct[road] = test_data
        # predict_df = pd.concat((predict_df, test_data[['crossroadID', 'index', 'flow']]), ignore_index=True)
        # test_data.set_index(['index', 'direction'], inplace=True)
        # return road, submit_df, test_data
        # submit_df[submit_df['crossroadID'] == road].loc[test_data.index, 'value'] = test_data['value']
        # submit_df.reset_index()[['date', 'crossroadID', 'direction', 'timeBegin', 'value']].\
        #     to_csv('./data/lr.csv', index=False)
    # predict_df.columns = ['crossroadID', 'timestamp', 'flow']
    for submit_road, train_road in pred_map.items():
        if train_road is not None:
            test_data = predict_dct[train_road].set_index(['index', 'direction'])
            s_index = submit_df.index & set(i + (submit_road, ) for i in test_data.index)
            test_index = list(i[:2] for i in s_index)
            submit_df.loc[s_index, 'value'] = test_data.loc[test_index, 'flow']
    for index in submit_df[submit_df['value'] == 0.1].index:
        submit_df.loc[index, 'value'] = submit_df.loc[index[0], 'value'].mean()
    submit_df = submit_df.reset_index()[['date', 'crossroadID', 'direction', 'timeBegin', 'value']]
    submit_df['value'] = submit_df['value'].apply(lambda x: int(x))
    submit_df.to_csv('./data/lr_bfs.csv', index=False)
    return submit_df


def timestamp_fmt(roadflow):
    roadflow['date'] = roadflow['timestamp'].apply(lambda x: '09-' + x.strftime('%d'))
    roadflow['timeBegin'] = roadflow['timestamp'].apply(lambda x: f'{x.hour}:{x.strftime("%M")}')
    roadflow['min_time'] = roadflow['timestamp'].apply(lambda x: int(x.strftime('%M')))
    pred = roadflow[roadflow['min_time'] >= 30]
    pred.drop(['timestamp'], axis=1, inplace=True)
    return pred


def result_fmt(term='first'):
    # 预测卡口与训练卡口邻接关系
    prp = PreProcessor(term)  # 数据管理器
    preMapping = get_testroad_adjoin(prp)
    submit_df = prp.get_submit()
    dire_dct = dict([road, list(set(df['direction']))] for (road, df) in submit_df.groupby('crossroadID'))
    submit_index, day_list = [], range(22, 26)  # 索引
    for day in day_list:
        submit_index.extend(pd.date_range(start=f'2019-09-{day} 07:00:00', periods=144, freq='5min'))
    train_index, day_list = [], range(15, 19)  # 索引
    for day in day_list:
        train_index.extend(pd.date_range(start=f'2019-09-{day} 07:00:00', periods=144, freq='5min'))
    train_index = list(str(i) for i in train_index)
    train_index_set = set(train_index)
    predict_df = pd.DataFrame(columns=['date', 'crossroadID', 'direction', 'timeBegin', 'value'])
    for pre_id in tqdm.tqdm(preMapping.keys()):
        instand_id = preMapping[pre_id]
        dire_list = dire_dct[pre_id].copy()
        if instand_id is None:
            roadflow = pd.DataFrame([predict_df['value'].mean()] * (144 * 4), columns=['value'])
            roadflow['timestamp'] = submit_index
            roadflow['crossroadID'] = pre_id
            pred = timestamp_fmt(roadflow)
        else:
            for dire, roadflow in prp.get_roadflow_by_road(instand_id):
                if not dire_list:
                    break  # 如空了则不要了，
                roadflow = roadflow.loc[list(set(roadflow.index) & train_index_set)]
                for ts in set(roadflow.index) ^ train_index_set:
                    roadflow[ts] = roadflow.mean()
                roadflow = pd.DataFrame(roadflow.values, columns=['value'])
                roadflow['timestamp'] = submit_index
                roadflow['crossroadID'] = pre_id
                pred = timestamp_fmt(roadflow)
                if prp.term:
                    pred['direction'] = dire_list.pop()
                predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
        while dire_list:  # 方向不够用的情况
            pred['direction'] = dire_list.pop()
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
    submit_time_set = set(submit_df['timeBegin'])
    predict_df.set_index('timeBegin', inplace=True)
    predict_df['value'].fillna(predict_df['value'].mean(), inplace=True)
    df = predict_df.loc[list(set(predict_df.index) & submit_time_set)].reset_index()[
        ['date', 'crossroadID', 'direction', 'timeBegin', 'value']]
    df.to_csv('./data/random.csv', index=False)
    return df


def arma_base(term='first'):
    submit_df = prp.get_submit()
    submit_index, day_list = [], range(22, 26)  # 索引
    for day in day_list:
        submit_index.extend(pd.date_range(start=f'2019-09-{day} 07:00:00', periods=144, freq='5min'))
    predict_df = pd.DataFrame()
    for instand_id in prp.load_buffer()['crossroadID'].unique():
        for dire, roadflow in prp.get_roadflow_by_road(instand_id):
            try:
                pred_pre = predict(roadflow)
            except Exception as e:
                print(instand_id, '\t', e)
                pred_pre = pd.Series([0.1] * (144 * 4), index=submit_index)
            # pred_pre = pred_pre.dropna(axis=0, how="any")
            pred_pre.fillna(pred_pre.mean(), inplace=True)
            for i in range(len(pred_pre)):
                pred_pre.iloc[[i]] = int(pred_pre.iloc[[i]])
            pred = pd.DataFrame(pred_pre.values, columns=['value'])
            pred['timestamp'] = submit_index
            pred['crossroadID'] = instand_id
            predict_df = pd.concat((predict_df, pred), axis=0, ignore_index=True)
    predict_df.to_csv('./data/aram_base.csv',index=False)
    return predict_df


if __name__ == '__main__':
    term = 'final'  # 初赛:first；复赛：final
    prp = PreProcessor(term)
    prp.dump_buffer(2)  # 载入数据
    prp.fill_na()  # 填入缺失值
    # arma_base(term)  # 时序模型
    # result_fmt(term)  # 随机模型
    # regression_many_x()  # 回归模型
    # regression_ex(term)  # 回归哦行
