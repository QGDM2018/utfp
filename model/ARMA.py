# from db_tools.base_table import Session,ZoneQuality_1
# from statsmodels.tsa.seasonal import  seasonal_decompose
# from statsmodels.tsa.arima_model import ARIMA
# import warnings
# warnings.filterwarnings('ignore')
# import json
# from datetime import datetime, timedelta
# import numpy as np
# import pandas as pd
# from settings import g_boundury,g_name
#
#
# # 读取参数
# try:
#     with open('params.json','r') as f:
#         params = json.load(f)
# except:
#     params = {}
# # 输入：七天数据
# # 路、或者区块
#
# # ****模型
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Simhei']
# plt.rcParams['axes.unicode_minus'] = False
#
# def accessMode(trueData, testData):
#     trueData = np.array(trueData).tolist()
#     testData = np.array(testData).tolist()
#     sum = 0
#     for i in range(len(trueData)):
#         sum += abs((trueData[i] - testData[i])/max(trueData[i],1e-9))
#     sum = 100*sum/len(trueData)
#     # print("预测模型评估(MAPE): {}%".format(round(sum, 3)))
#     # print("评估标准：越接近0，模型预测越接近实际值")
#     return sum
# class ModeDecomp(object):
#     def __init__(self, dataSet,type, test_size = 8):
#         # data = dataSet.set_index('date')
#         # data.index = pd.to_datetime(data.index)
#         self.dataSet = dataSet.astype(float)
#         self.test_size = test_size
#         self.train_size = len(self.dataSet) - self.test_size
#         self.train = self.dataSet[type][:len(self.dataSet) - test_size]
#         self.train = self._diff_smooth(self.train)
#         # self.train.plot()
#         # plt.show()
#         # input(self.train)
#         # input("平滑")
#         self.test =  self.dataSet[type][-test_size:]
#
#     # 对数据进行平滑处理
#     def _diff_smooth(self, dataSet):
#         dif = dataSet.diff()         # 差分序列
#         td = dif.describe()
#         high = td['75%'] + 1.5 * (td['75%'] - td['25%'])  # 定义高点阈值，1.5倍四分位距之外
#         low = td['25%'] - 1.5 * (td['75%'] - td['25%'])  # 定义低点阈值，同上
#         # 变化幅度超过阈值的点的索引
#         forbid_index = dif[(dif > high) | (dif < low)].index
#         # dataSet.loc[forbid_index] = dataSet.mean()
#         i = 0
#         while i < len(forbid_index) - 1:
#             n = 1  # 发现连续多少个点变化幅度过大，大部分只有单个点
#             start = forbid_index[i]  # 异常点的起始索引
#             while forbid_index[i + n] == start + timedelta(minutes=60*n):
#                 n += 1
#                 if (i + n) > len(forbid_index) - 1:
#                     break
#             i += n - 1
#             end = forbid_index[i]  # 异常点的结束索引
#             # 用前后值的中间值均匀填充
#             try:
#                 value = np.linspace(dataSet[(start - timedelta(hours=1))], dataSet[(end + timedelta(hours=1))], n)
#                 dataSet[start: end] = value
#             except:
#                 pass
#             i += 1
#         return dataSet
#
#     def decomp(self, freq):
#         decomposition = seasonal_decompose(self.train, freq=freq, two_sided=False)
#         self.trend = decomposition.trend
#         self.seasonal = decomposition.seasonal
#         self.residual = decomposition.resid
#         # decomposition.plot()
#         # plt.show()
#         # # input()
#         d = self.residual.describe()
#         delta = d['75%'] - d['25%']
#         self.low_error, self.high_error = (d['25%'] - 1*delta, d['75%'] + 1*delta)
#
#     def trend_model(self, order):
#         self.trend.dropna(inplace=True)
#         self.trend_model_ = ARIMA(self.trend, order).fit(disp=-1, method='css')
#         # return self.trend_model_
#
#     def predict_new(self):
#         """
#         预测新数据
#         :return:
#         """
#         n = self.test_size
#         self.pred_time_index = pd.date_range(start=self.train.index[-1], periods = n+1, freq='60min')[1:]
#         self.trend_pred = self.trend_model_.forecast(n)[0]
#         pred_time_index = self.add_season()
#         return pred_time_index
#
#     def add_season(self):
#         '''
#         为预测出的趋势数据添加周期数据和残差数据
#         '''
#         self.train_season = self.seasonal[:self.train_size]
#         values = []
#         low_conf_values = []
#         high_conf_values = []
#
#         for i,t in enumerate(self.pred_time_index):
#             trend_part = self.trend_pred[i]
#             #相同时间的数据均值
#             season_part = self.train_season[
#                 self.train_season.index.time == t.time()
#                 ].mean()
#             #趋势+周期+误差界限
#             predict = trend_part + season_part
#             low_bound = trend_part + season_part + self.low_error
#             high_bound = trend_part + season_part + self.high_error
#
#             values.append(predict)
#             low_conf_values.append(low_bound)
#             high_conf_values.append(high_bound)
#         self.final_pred = pd.Series(values, index=self.pred_time_index, name='predict')
#         self.low_conf = pd.Series(low_conf_values, index=self.pred_time_index, name='low_conf')
#         self.high_conf = pd.Series(high_conf_values, index=self.pred_time_index, name='high_conf')
#
#         return self.pred_time_index
#
#
#
#
# # *****加载数据集
# # 获取七天的数据
# def get_data_1(day,zone):
#     day = datetime.strptime(day[:10], "%Y-%m-%d")
#     end_day = (day + timedelta(days=2)).timestamp()   #一周的时间
#     day = (day -timedelta(days=5)).timestamp()
#     s = Session()
#     d_data = {}     #  用字典来分段
#     qry = s.query(ZoneQuality_1)
#     for rst in qry.filter(ZoneQuality_1.g_zone==zone).all():
#         date_time = rst.date_time
#         if (day <= date_time.timestamp() <= end_day):
#             # hour = date_time.hour
#             # delta =  2 - hour%4
#             # date_time = date_time - timedelta(days=1)
#             if date_time in d_data:
#                 a_time,dens,flow =  d_data[date_time]
#                 d_data[date_time] = [(a_time+rst.average_speed)/2,(dens+rst.density)/2,
#                                      (flow+rst.flow)/2]
#             else:
#                 d_data[date_time] = [rst.average_speed,rst.density,rst.flow]
#     df = pd.DataFrame(d_data).T
#     df.columns = ['average_speed','density','flow']
#     return df
# def get_data(day,zone):
#     end_day = (day + timedelta(days=7)).timestamp()   #一周的时间
#     day = day.timestamp()
#     s = Session()
#     d_data = {}     #  用字典来分段
#     for rst in s.query(ZoneQuality).filter(ZoneQuality.g_zone==zone).all():
#         date_time = rst.date_time
#         if (day <= date_time.timestamp() <= end_day):
#             hour = date_time.hour
#             delta =  2 - hour%4
#             date_time = date_time + timedelta(hours=delta)
#             if date_time in d_data:
#                 a_time,dens,flow =  d_data[date_time]
#                 d_data[date_time] = [(a_time+rst.average_speed)/2,(dens+rst.density)/2,
#                                      (flow+rst.flow)/2]
#             else:
#                 d_data[date_time] = [rst.average_speed,rst.density,rst.flow]
#     df = pd.DataFrame(d_data).T
#     df.columns = ['average_speed','density','flow']
#     return df
#
# def draw():
#     pass
#         # 开始训练
#     # print(best_i,best_j,best_k)
#     # mode.trend_model(order=(best_i, best_j, best_k))
#     # pred_time_index = mode.predict_new()
#     # pred = mode.final_pred
#     # test = mode.test
#     # sum = accessMode(test, pred)
#     # print("预测模型评估(MAPE): {}%".format(round(sum, 3)))
#     #
#     # # ***画图部分
#     # plt.plot(mode.train)
#     # plt.subplot(212)
#     # test1 = np.array(test).tolist()
#     # test = pd.Series(test1, index=pred_time_index, name='test')
#     # pred.plot(color='salmon', label='Predict')
#     # test.plot(color='steelblue', label='Original')
#     # mode.low_conf.plot(color='grey', label='low')
#     # mode.high_conf.plot(color='grey', label='high')
#     # plt.legend(loc='right')
#     # plt.tight_layout()
#     # plt.show()
#
# # *****建模部分
# def predict_(data,type):
#     size = 48
#     mode = ModeDecomp(data, type, test_size=size)
#     mode.decomp(size)
#     sum_min = 1000
#     best_i,best_j,best_k = 0,0,1
#     max_iter = 10
#     while max_iter:
#         max_iter -= 1
#         i,k = np.random.choice(10,2)
#         j = np.random.choice(2,1)[0]
#         # print(i,j,k)
#         try:
#             mode.trend_model(order=(i, j, k))
#             pred_time_index = mode.predict_new()
#             pred = mode.final_pred
#             test = mode.test
#             sum = accessMode(test, pred)
#             # print(sum)
#             if sum < sum_min:
#                 sum_min = sum
#                 # print(i,j,k)
#                 best_i = i
#                 best_j = j
#                 best_k = k
#         except Exception as e:
#             # print(e)
#             pass
#     return mode,best_i,best_j,best_k
#
# def predict(data,type,params=None):
#     size = 48
#     if params is not None:
#         best_i, best_j, best_k = params
#         mode = ModeDecomp(data, type, test_size=size)
#         mode.decomp(size)
#     else:
#         mode,best_i, best_j, best_k = predict_(data,type)
#     print(best_i,best_j,best_k)
#     # 开始训练
#     # mode.trend_model(order=(best_i, best_j, best_k))
#     # pred = mode.final_pred
#     # pred = np.array(pred).tolist()
#     # train = np.array(mode.train).tolist()
#     # y = [train , pred]
#     # x = [[str(time)[5:10] for time in data.index[:(len(data)-size)]], [str(time)[5:10] for time in data.index[(len(data)-size):]]]
#
#     try:
#         mode.trend_model(order=(best_i, best_j, best_k))
#     except Exception as e:
#         print(e)
#         mode, best_i, best_j, best_k = predict_(data, type)
#         mode.trend_model(order=(best_i, best_j, best_k))
#     pred_time_index = mode.predict_new()
#     pred = mode.final_pred
#     test = mode.test
#     sum = accessMode(test, pred)
#     print("预测模型评估(MAPE): {}%".format(round(sum, 3)))
#
#     # # ***画图部分
#     # plt.plot(mode.train)
#     # plt.subplot(212)
#     # test1 = np.array(test).tolist()
#     # test = pd.Series(test1, index=pred_time_index, name='test')
#     # pred.plot(color='salmon', label='Predict')
#     # test.plot(color='steelblue', label='Original')
#     # mode.low_conf.plot(color='grey', label='low')
#     # mode.high_conf.plot(color='grey', label='high')
#     # plt.legend(loc='right')
#     # plt.tight_layout()
#     # plt.show()
#     train = np.array(mode.train).tolist()
#     y = [train , pred]
#     x = [[str(time)[5:10] for time in data.index[:(len(data)-size)]], [str(time)[5:10] for time in data.index[(len(data)-size):]]]
#     return [x,y]
# def fit(data,ctype):
#     size = 12
#     mode = ModeDecomp(data, ctype, test_size=size)
#     mode.decomp(size)
#     best_i,best_j,best_k = 0,0,1
#     sum_min = 1000
#     max_iter = 50
#     while max_iter:
#         max_iter -= 1
#         i,k = np.random.choice(10,2)
#         j = np.random.choice(2,1)[0]
#         # print(i,j,k)
#         try:
#             mode.trend_model(order=(i, j, k))
#             pred_time_index = mode.predict_new()
#             pred = mode.final_pred
#             test = mode.test
#             sum = accessMode(test, pred)
#             if sum < sum_min:
#                 sum_min = sum
#                 best_i = i
#                 best_j = j
#                 best_k = k
#         except Exception as e:
#             # print(e)
#             pass
#     return int(best_i),int(best_j),int(best_k)
# def road_quality_fit():
#     # 对每一天，每个区域
#     params = {}
#     try:
#         for d in range(10,20)[:1]:
#             day = str(datetime.strptime("2017-01-31","%Y-%m-%d") + timedelta(days=d))
#             params[day[:10]] = {}
#             dic = params[day[:10]]
#             for z in range(1,12)[:1]:
#                 print("day---",day,"zone--",z)
#                 df = get_data_1(day,z)
#                 dic[z] = {}
#                 dic_ = dic[z]
#                 for c in df.columns:
#                         # 异常值 处理
#                     if c == 'average_speed':
#                         col = df[c]
#                         df.loc[:, c][col < .5] *= 100
#                         df.loc[:, c][(col > 0.5) & (col < 10)] *= 10
#                         df.loc[:, c][col > 100] /= 10
#                     elif c == 'density':
#                         col = df[c]
#                         df.loc[:, c][col < 10] *= 100
#                         df.loc[:, c][col > 1e5] /= 10
#                     dic_[c] = fit(df[[c]],c)
#     except Exception as e:
#         print(e)
#     with open("params.json",'w') as f:
#         json.dump(params,f)
#     return params
#
#
# # ****预测部分
# def road_quality_anylyse(day,zone):
#     '''区块'''
#     df = get_data_1(day,zone)
#     # 数据预处理
#     for c in df.columns:
#         # 异常值 处理
#
#         if c == 'average_speed':
#             col = df[c]
#             meanV = col.median()
#             df.loc[:,c][col <.5 ] = meanV #*= 100
#             df.loc[:, c][(col> 0.5) & (col <10)] = meanV#*= 10
#             df.loc[:, c][col > 100] =meanV# /= 10
#         elif c == 'density':
#             col = df[c]
#             df.loc[:,c][col <10 ] *= 100
#             df.loc[:, c][col > 1e5] /= 10
#     name = ("道路平均速度","道路车辆密度","道路车流量")
#     rst = {}
#     param = params[day][str(zone)]
#     for i,c in enumerate(('average_speed','density','flow')):
#         data = df[[c]]
#         rst[c] = {'type':g_name[zone]+name[i]}
#         x,y = predict(data[[c]],c,param[c])   #[x,y]
#         rst[c]['x'] = x
#         rst[c]['y'] = y
#     return rst
#
#
#
# if __name__ =='__main__':
#     # a = road_quality_anylyse('2017-02-10',4)
#     a = road_quality_anylyse("2017-02-10",1)
#     # a = road_quality_fit()
