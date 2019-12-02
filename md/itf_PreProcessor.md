获取单个路口各个时段车流量数据

```python
    def get_roadFlow(self,i) -> object:
        '''获取单个路口各个时段车流量数据
        :param i:天数
        :return:
            dfFlow：pd.DataFrame 原始车流数据表
            dFlow: {crossroadID:pd.Series} 车流量时序数据
        '''

# 调用示例
from pre_process import PreProcessor
prp = PreProcessor()    # 数据管理器
dfFlow,dFlow =prp.get_flow(1)	# 原始车流数据表，车流量时序数据
```

