#### 邻接数据集

​	提取卡口某一时刻的流量(flow) 和其邻接卡口流量的均值(mean_flow)

作为目标值

```python
crossroadID, timestamp, flow -> mean_flow, flow
```

​	缺失值：drop；

**思路**：利用多级索引 + 表格转置，得到列名为卡口的表格，方便索引

```python
crossroadID,direction , timestamp, flow ->
crossroadID, timestamp, flow ->
					crossroadID1, crossroadID2, crossroadID3
timestamp direction												-> 
```

方式一：按行遍历，依次用路口**某时刻**的流量构建新数据

方式二：按列遍历，借用矩阵运算依次用路口**所有**流量构建新数据



问题：

1. 流量数据的一些卡口，没有出现在路网中

2. ```python
   pd.DataFrame([[1,pd.np.nan,1]]).mean(axis=1)
   # None是空,不算它
   ```

#### 邻接数据集
提取卡口某一时刻各个方向的流量(flow) 和其邻接卡口流量的各个方向的流量（X）

```python
crossroadID, timestamp, flow -> (road1, direciont1), (road2, directino2), direction, flow
```


选择：

1. pd.DataFrame.append

2. pd.concate
