import pandas as pd
from ast import literal_eval

# 读取数据集
data = pd.read_csv('../dataset/train_unknown.csv')
data2 = pd.read_csv('../dataset/train_known.csv')

# 获取时间戳列的最小值
min_timestamp = data['common_ts'].min()

# 将时间戳列中的每个值都减去最小值，得到相对时间差
data['common_ts'] = data['common_ts'] - min_timestamp

min_timestamp = data2['common_ts'].min()

data2['common_ts'] = data2['common_ts'] - min_timestamp

# 保存处理后的数据集到新文件
data.to_csv('../dataset/train_unknown_cutTime.csv', index=False)
data2.to_csv('../dataset/train_known_cutTime.csv', index=False)