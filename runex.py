from pre_process import PreProcessor, get_testroad_adjoin, pd, np
import matplotlib.pyplot as plt
from model.ARMA import predict
from feature_en import FeatureEn
import tqdm
from model.AP import ap_predict

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


def arma_ex(term='first'):
    prp = PreProcessor(term)  # 数据管理器
    preMapping = get_testroad_adjoin(prp)
    for pre_id in tqdm.tqdm(preMapping.keys()):
        instand_id = preMapping[pre_id]
        try:
            pred_pre = predict(prp.get_roadflow_by_road(instand_id))
        except Exception as e:
            print(instand_id, end='\t')
            print(e)
            continue
        pred_pre.to_frame()
        pred_pre = pred_pre.dropna(axis=0, how="any")
        for i in range(len(pred_pre)):
            pred_pre.iloc[[i]] = int(pred_pre.iloc[[i]])
        pred = pd.DataFrame(pred_pre.values, columns=['value'])
        pred['timestamp'] = pred_pre.index
        pred['date'] = pred['timestamp'].apply(lambda x: x.strftime('%d'))
        pred['timeBegin'] = pred['timestamp'].apply(lambda x: x.strftime('%H:%M'))
        pred['crossroadID'] = instand_id
        pred['min_time'] = pred['timestamp'].apply(lambda x: int(x.strftime('%M')))
        pred = pred[pred['min_time'] >= 30]
        pred.drop(['timestamp'], axis=1, inplace=True)
        order = ['date', 'crossroadID', 'timeBegin', 'value']
        pred = pred[order]
        pred.to_csv(r'data\tmp\{}\pred_{}.csv'.format(term, pre_id), index=False,
                    columns=['date', 'crossroadID', 'timeBegin', 'value'])


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


def regression_ex_l(term='first'):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    fe = FeatureEn(term)
    data = pd.DataFrame(fe.extract_adjoin_by_col())
    r2_rst = {}
    for road, groups in data.groupby('crossroadID'):
        X_train, X_test, y_train, y_test = train_test_split(groups[['mean_flow']], groups['flow'], test_size=.33)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        r2_rst[road] = r2_score(y_test, lr.predict(X_test))
    return r2_rst
    # 对整个数据集进行回归

    # 对单个路口进行回归

    # return a


if __name__ == '__main__':
    term = 'final'  # 初赛:first；复赛：final
    # term = 'first'  # 初赛:first；复赛：final
    # arma_ex(term)  # 时序模型
    a = FeatureEn(term).extract_adjoin_by_col()




