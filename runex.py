from pre_process import PreProcessor,pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

# ****载入数据
day = 3
prp = PreProcessor()    # 数据管理器
# for i in range(1,24):
#     dfFlow,dFlow =prp.get_flow(i)	# 原始车流数据表，车流量时序数据 n
dfFlow,dFlow =prp.get_roadFlow(day)	# 原始车流数据表，车流量时序数据 n

# *****绘图示例
# key = list(dFlow.keys())[0]
# seFolw =dFlow[key]
# seFolw.plot()
# plt.title(f'{day}号交通口{key}车流量时序图')
# plt.ylabel('车流量/5min')
# plt.xlabel('时间/t')
# plt.show()