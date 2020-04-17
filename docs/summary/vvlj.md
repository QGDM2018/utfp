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

+ 



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

​	