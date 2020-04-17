此文件仅对此比赛的数据的构成做出说明。

一、数据内容包含两类：
	1. 青岛交通卡口的过车数据（主数据）
	2. 青岛出租车的GPS记录（辅助数据）

二、数据时间：
	2019年8月1日~2019年8月23日

三、训练集和测试集划分：
	按照日期划分。
		训练集：8月1日~8月19日
		测试集：8月20日~8月23日

四、文件夹名字说明：
	1. trainTaxiGPS：训练集对应日期的出租车GPS记录
	2. testTaxiGPS：测试集对应日期的出租车GPS的测试数据
	3. trainCrossroadFlow：训练集对应日期的交通卡口过车数据
		此文件夹下，额外包含
			路网信息 roadnet.csv
			交通卡口名字信息 crossroadName.csv
	4. testCrossroadFlow：测试集对应日期、特定时段的交通卡口过车数据
		此文件夹下，额外包含
			提交结果的样例文件 submit_example.csv
	5. 字段对应关系说明.xlsx：为原始的交通道路描述信息，供参考。基于此文件生成了roadnet.csv和crossroadName.csv
	