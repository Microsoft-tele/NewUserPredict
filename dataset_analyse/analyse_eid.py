import pandas as pd
from tools.config_file import NewUserPredictParams
params = NewUserPredictParams()

# 读取CSV文件
train_csv = pd.read_csv(params.test_csv)
# 使用value_counts统计每个eid对应的udmap数量
eid_udmap_counts = train_csv['eid'].value_counts()
# 输出每个eid对应的udmap数量统计信息
print(eid_udmap_counts)
print(train_csv)
