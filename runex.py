from pre_process import PreProcessor
prp = PreProcessor()    # 数据管理器


'direction,laneID,timestamp,crossroadID,vehicleID'

train,test = prp.load_csv()
