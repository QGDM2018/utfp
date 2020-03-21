from pre_process import PreProcessor, dump_buffer # saving,trans_1,trans
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


def plot_roadflow():
    # ******载入数据******
    day = 3
    prp = PreProcessor()  # 数据管理器
    dfFlow, dFlow =prp.get_roadflow(day)	# 原始车流数据表，车流量时序数据
    # *****绘图示例******
    key = list(dFlow.keys())[0]
    seFolw =dFlow[key]
    seFolw.plot()
    plt.title(f'{day}号交通口{key}车流量时序图')
    plt.ylabel('车流量/5min')
    plt.xlabel('时间/t')
    plt.show()


if __name__ == '__main__':
    dump_buffer()  # 缓存
