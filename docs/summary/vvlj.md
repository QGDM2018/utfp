[TOC]



### 关于协作

由于git能够追溯文件中某个位置的修改，多个人维护同一个文件容易产生冲突。

+ 因此当多个人负责同一个功能的不同部分时，如预处理或者特征工程，尽量分文件进行协作，或者讨论一致后再push上去，避免冲突。
+ 利用高级指令



### 知识点积累

#### 时间(datetime)

+ 时间戳加减

```python
# timedelta类型可用于时间加减运算
from datetime import timedelta
timedelta(days=1, minutes=1, seconds=1)
```

+ 字符串转时间戳

```python
# str -> pandas._libs.tslibs.timestamps.Timestamp
import pandas as pd
pd.to_datetime('2019-09-25 18:00:00')
```





#### DataFrame

+ 分组

```python
# name, group_df(含分组的字段)
for name, group_df in df.groupby(['column', ]):
    pass

# name, group_index(索引)
for name, group_index in df.groupby(['column', ]).groups.items():  # .groups是dict
    pass

# group_df(含分组的字段)
group_df = df.groupby(['column', ]).get_group(name)
```

+ apply，按列迭代

```python
# Series，第一个参数x便是series的元素；可以传入额外参数如y
series.apply(lambda x, y: x + y, y = 1)

# DataFrame, column1的元素是x，column2的元素是y，不可以传入额外参数
df['column1', 'column2'].apply(lambda x, y: x + y)
```

+ 去重

```python
# subset: 去重的列； keep：last，保留最后一个出现，默认第一个出现
df.drop_duplicates(subset=['column', ], keep='last')
```

+ 索引

```python
# 设置某列为列索引
df.set_index(['column', ])
# 将列索引还原成column
df.reset_index()
# 将列索引设置成行索引
df.unstack()	# 顺序默认从最内部的索引开始设置

# 行多级索引
df['column1']['column2']	# 从上到下依次索引
df['column1', 'column2']	# 多级索引
# 列多级索引
df.loc['index1'].loc['index2']	# 从左到右依次索引
df.loc[('index1', 'index2')]	# 多级索引
```

+ 连接表格

```python
# 向DataFrame添加多行，利用纵向表格连接比单行添加快
# axis=0,纵向连接；ignore_index，重新编号索引
pd.concat((df1, df2), axis=0, ignore_index=True)
# axis=1,横向连接
pd.concat((df1, df2), axis=0, ignore_index=True)
```



#### csv文件写入

```python
with open(path, 'w', newline='') as f:
	handler = csv.writer(f)
	handler.writerow([[]])
```



#### 字符串控制

+ 右靠齐补零

```python
'7:30'.rjust(5, "0")
```



### 总结

​	这次项目断断续续做了很久，从19年11月到20年4月，整整有5个月的时间，不过真正花在上面的时间并没有那么多。初赛到复赛之间有一段数据筹备的时间。这是第一次正式打比赛，虽然名次不是很高，但是还是收获到很多东西的，包括团队间的合作、开始一个数据挖掘项目的注意事项。

​	首先是团队合作，一个团队一起从事一个项目时，要注意每个人的分工以及进度。分工是按照数据挖掘的流程进行分工，如预处理、特征工程和建模。每个人不一定要了解其他人的进度，但一定要了解项目整体走到哪一步了，比如大家是否清楚题目的要求是什么，是否对数据有一个基本的认识，是否了解一些基础函数的作用。人通常是有惰性的，为了提高协作效率，就要多多主动去了解其他人的进度。另外，文档、代码的规范化也没有做好，这就给了沟通和记录加大了难度。

​	其次是对这次数据挖掘的体会。弄清楚数据的概况很重要，例如缺失值、含义等等。由于前期没有了解好数据，后期再对模型进行修修补补就很麻烦。本次交通流数据有很多缺失的情况，例子流量的缺失、邻接卡口的缺失、甚至测试集整个数据集的缺失。这些缺失导致了很多设想好的模型不可用，例如回归模型不可用，即利用岭回归来确定某个卡口的其临界卡口不同车流量的关系，权值越高意味着更加可能两特定方向的路口相连，最后利用邻接关系和这些权值求出测试卡口的流量。下次进行比赛必须花更多的时间阅读数据的描述，以及比赛的要求。

​	最后，每次项目后，需要多思考从项目中学到了什么，有什么不足，或多或少都写一点。

