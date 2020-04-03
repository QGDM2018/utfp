### 原始数据

​	某**时刻**通过某**卡口**的**车辆**

```python
direction, laneID, timestamp, crossroadID, vehicleID
```



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
