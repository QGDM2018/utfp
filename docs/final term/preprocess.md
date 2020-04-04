### 原始数据

+ 某**时刻**通过某**卡口**的**车辆**

```python
direction, laneID, timestamp, crossroadID, vehicleID
```
+ 训练集、测试集与submit文件
  + 训练集： 1-19号数据，包含整天的数据 -- 07:00, 07:05, ..., 18:55
  + 测试集： 22-25号，包含一天中前半个钟的数据 -- 07:00, 07:05, ..., 07.25, 08:00, ..., 18:25
  + submit文件： 22-25号，只需要预测前后个钟数据 -- 07:30, 07:55, ..., 18:55


### 需求

​	统计卡口每5min内的车流量。

1. 单个卡口的流量时序数据
2. 单个时段所有卡口的流量数据



### 处理方式

1. 以字典的形式pickle进文件缓存

```python
{'road': pd.Series(flow_lst, index=timestamp)}	# 初赛
{'road': 'direction': pd.Series(flow_lst, index=timestamp)}  # 复赛
```

2. 以表格的形式存入文件缓存
   1. 读取原始文件
   2. 统计卡口每5min内的车流量数据
   3. 按批写入文件

```python
timestamp, crossroadID, flow  # 初赛
direction, timestamp, crossroadID, flow  # 复赛
```

### 接口
```python
# 若报错则先确认data目录结构
# 再调用PreProcessor('final').dump_buffer()缓存数据
flow_df = PreProcessor('final').get_timeflow() 
```