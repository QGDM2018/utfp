from pre_process.pre_process import PreProcessor
import pre_process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import warnings
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')


class ModeDecomp(object):
    def __init__(self, dataSet, test_data, test_size = 24):
        data = dataSet.set_index('timestamp')
        data.index = pd.to_datetime(data.index)
        self.dataSet = data
        self.test_size = test_size
        self.train_size = len(self.dataSet)
        self.train = self.dataSet['flow']
        self.train = self._diff_smooth(self.train)
        self.test = test_data['flow']

    # 对数据进行平滑处理
    def _diff_smooth(self, dataSet):
        dif = dataSet.diff()         # 差分序列
        td = dif.describe()

        high = td['75%'] + 1.5 * (td['75%'] - td['25%'])  # 定义高点阈值，1.5倍四分位距之外
        low = td['25%'] - 1.5 * (td['75%'] - td['25%'])  # 定义低点阈值，同上

        # 变化幅度超过阈值的点的索引
        forbid_index = dif[(dif > high) | (dif < low)].index
        i = 0
        while i < len(forbid_index) - 1:
            n = 1  # 发现连续多少个点变化幅度过大，大部分只有单个点
            start = forbid_index[i]  # 异常点的起始索引
            while forbid_index[i + n] == start + timedelta(minutes=60*n):
                n += 1
                if (i + n) > len(forbid_index) - 1:
                    break
            i += n - 1
            end = forbid_index[i]  # 异常点的结束索引
            # 用前后值的中间值均匀填充
            try:
                value = np.linspace(dataSet[start - timedelta(minutes=60)], dataSet[end + timedelta(minutes=60)], n)
                dataSet[start: end] = value
            except:
                pass
            i += 1
        return dataSet

    def decomp(self, freq):
        decomposition = seasonal_decompose(self.train, freq=freq, two_sided=False)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        # decomposition.plot()
        # plt.show()
        d = self.residual.describe()
        delta = d['75%'] - d['25%']
        self.low_error, self.high_error = (d['25%'] - 1*delta, d['75%'] + 1*delta)

    def trend_model(self, order):
        self.trend.dropna(inplace=True)
        self.trend_model_ = ARIMA(self.trend, order).fit(disp=-1, method='css')
        # return self.trend_model_

    def predict_new(self):
        """
        预测新数据
        :return:
        """
        n = self.test_size

        self.pred_time_index = pd.date_range(start=self.train.index[-1], periods = n+1, freq='5min')[1:]
        self.trend_pred = self.trend_model_.forecast(n)[0]
        pred_time_index = self.add_season()
        return pred_time_index



    def add_season(self):
        '''
        为预测出的趋势数据添加周期数据和残差数据
        '''
        self.train_season = self.seasonal[:self.train_size]
        values = []
        low_conf_values = []
        high_conf_values = []

        for i, t in enumerate(self.pred_time_index):
            trend_part = self.trend_pred[i]
            # 相同时间的数据均值
            season_part = self.train_season[
                self.train_season.index.time == t.time()
                ].mean()
            # 趋势+周期+误差界限
            predict = trend_part + season_part
            low_bound = trend_part + season_part + self.low_error
            high_bound = trend_part + season_part + self.high_error

            values.append(predict)
            low_conf_values.append(low_bound)
            high_conf_values.append(high_bound)
        self.final_pred = pd.Series(values, index=self.pred_time_index, name='predict')
        self.low_conf = pd.Series(low_conf_values, index=self.pred_time_index, name='low_conf')
        self.high_conf = pd.Series(high_conf_values, index=self.pred_time_index, name='high_conf')

        return self.pred_time_index


def predict(X):
    dataSet = X[:-144]
    # input(len(dataSet))
    a = 288 * 4
    test_data = np.zeros(a)
    test_data = pd.DataFrame(test_data, columns=['flow'])
    data = pd.DataFrame(dataSet.values, columns=['flow'])
    data['timestamp'] = dataSet.index
    size = 288 * 4
    mode = ModeDecomp(data, test_data, test_size=size)
    mode.decomp(size)
    for lis in [[3, 1, 3], [1, 2, 3], [5, 2, 3], [1, 1, 2], [3, 1, 4], [0, 0, 1]]:
        try:
            mode.trend_model(order=(lis[0], lis[1], lis[2]))
            break
        except:
            continue
    # mode.trend_model(order=(0, 0, 1))
    pred_time_index = mode.predict_new()
    pred = mode.final_pred
    test = mode.test
    # insert_Operateefficient_predict(str(area), str(Date), str(paramster[0]), str(paramster[1]), str(paramster[2]))
    # plt.subplot(211)
    # plt.plot(mode.train)
    # plt.subplot(212)
    # test1 = np.array(test).tolist()
    # test = pd.Series(test1, index=pred_time_index, name='test')
    # pred.plot(color='salmon', label='Predict')
    # test.plot(color='steelblue', label='Original')
    # mode.low_conf.plot(color='grey', label='low')
    # mode.high_conf.plot(color='grey', label='high')
    # plt.legend(loc='right')
    # plt.tight_layout()
    # plt.show()
    # accessMode(test, pred)

    return pred


def create_test_data():
    test_data = pd.read_csv('data/testCrossroadFlow/submit_example.csv')
    for i in range(len(test_data)):
        retail_data = test_data.iloc[[i]]
        date = retail_data['date'][i]
        crossroadID = retail_data['crossroadID'][i]
        timeBegin = retail_data['timeBegin'][i]

        date = 20
        crossroadID = 100001
        open_file = 'data/tmp/pred_{}_{}.csv'.format(date, crossroadID)
        pred_data = pd.read_csv(open_file,header=0,index_col=0)
        if len(timeBegin) == 4:
            search_time = '2019-08-'+str(date)+" 0"+timeBegin+":00"
        elif len(timeBegin)==5:
            search_time = '2019-08-' + str(date) + " " + timeBegin + ":00"
        pred_flow = pred_data.loc[pred_data['timestamp'] == search_time]['flow'].values[0]
        test_data.iloc[[i]]['value'] = pred_flow
        pred_flow = pred_data[search_time]['flow']


# # 画图
# if __name__ == '__main__':
#     day = 3
#     prp = PreProcessor()
#     preMapping = pre_process.saving(prp)
#     print(preMapping)
#     for pre_id in preMapping.keys():
#         try:
#             instand_id = preMapping[pre_id]
#             pred = predict(instand_id)
#             store_data(pred, pre_id)
#         except:
#             continue

