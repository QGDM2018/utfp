import pandas as pd
class PreProcessor:
    def __init__(self):
        self.load_csv()

    def load_csv(self):
        '''从文件夹中载入csv表格'''
        dirpath = 'data/'
        encoding = 'utf-8'
        self.test = pd.read_csv(dirpath+'testCrossroadFlow/test_trafficFlow_20.csv',encoding=encoding)    # 读取20号的
        self.train = pd.read_csv(dirpath+'trainCrossroadFlow/train_trafficFlow_1.csv',encoding=encoding)   # 读取1号的
        return self.test,self.train

    def cal_flow(self,data):
        '''计算流量'''

