from pre_process import PreProcessor,pd,saving
from multiprocessing.pool import Pool
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

def done():
    freeze_support()
    # ******载入数据******
    day = 3
    prp = PreProcessor()  # 数据管理器
    sPredRoad, mapping = saving(prp)
    # for i in range(1,24):
    #     dfFlow,dFlow =prp.get_roadFlow(i)	# 原始车流数据表，车流量时序数据 n
    # dfFlow,dFlow =prp.get_roadFlow(day)	# 原始车流数据表，车流量时序数据 n
    # dfFlow,dFlow =prp.get_timeFlow(day)	# 原始车流数据表，车流量时序数据 n

    # ******缓存数据******
    # from os import walk
    # visited = set()
    # for _,_,files in walk(r'E:\数据挖掘项目\qgTask\utfp\data\tmp'):
    #     for f in files:
    #         if 'road' in f:
    #             visited.add(f.split('_')[1])
    # lRoadId = set(prp.load_csv(1)['crossroadID'])
    # lRoadId ^= visited
    # pool = Pool(4)
    # pool.map(prp.get_roadFlow_total,lRoadId)

    # roadFlow = prp.get_roadFlow_total(100001)
    # *****绘图示例******
    # key = list(dFlow.keys())[0]
    # seFolw =dFlow[key]
    # seFolw.plot()
    # plt.title(f'{day}号交通口{key}车流量时序图')
    # plt.ylabel('车流量/5min')
    # plt.xlabel('时间/t')
    # plt.show()

if __name__ == '__main__':
    day = 3
    prp = PreProcessor()  # 数据管理器
    predMapping = saving(prp)