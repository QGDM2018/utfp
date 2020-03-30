from pre_process import PreProcessor, get_testroad_adjoin, pd
import matplotlib.pyplot as plt
from model.ARMA import predict
from feature_en import FeatureEn
import tqdm
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


if __name__ == '__main__':
    term = 'final'  # 初赛:first；复赛：final
    # term = 'first'  # 初赛:first；复赛：final
    PreProcessor(term).dump_buffer(2)  # 调用一次即可
    arma_ex(term)  # 时序模型


