# utfp
城市交通流量时空预测---山东省数据应用（青岛）创新创业大赛。http://sdac.qingdao.gov.cn/common/cmptIndex.html



#### 预处理说明

​	原始数据没有流量的信息，通过预处理函数来统计卡口每5min内的车流量。

为了能够顺利调用预处理函数，需要统一data的目录结构（需**手动创建**）。项目的data目录未上传，结构如下；其中first是初赛数据、final是复赛数据

+ data
  
  + first
  
    + testCrossroadFlow
    + testTaxiGPS
  
    + trainCrossroadFlow
    + trainTaxiGPS
  
  + final
  
    + test_user
    + train
    
  + tmp
  
    + first
    + final
  
    

**调用示例**

```python
# 由于相对路径问题，需要在项目根目录调用，如runex.py
term = 'final'  # 初赛:first；复赛：final
process_num = 2  # 进程数量
PreProcessor(term).dump_buffer(process_num)

# 得到的流量文件如下
# 初赛数据：./data/0_flow_data.csv  ['crossroadID', 'timestamp', 'flow']
# 复赛数据: ./data/1_flow_data.csv  ['crossroadID', 'direction', 'timestamp', 'flow']
```

