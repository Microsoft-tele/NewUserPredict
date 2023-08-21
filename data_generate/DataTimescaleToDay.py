# 这个文件是把数据集里的时间戳，转换为只考虑日子day和小时hour的两列

import pandas as pd
from datetime import datetime

# 读取CSV文件
data = pd.read_csv('../dataset/train_unknown.csv')


# 将时间戳列转换为日期和时间格式
data['common_ts'] = pd.to_datetime(data['common_ts'], unit='ms')

# 将时间戳列转换为日期和时间格式
data['common_ts'] = pd.to_datetime(data['common_ts'], unit='ms')

# 提取日期和小时
data['day'] = data['common_ts'].dt.day
data['time'] = data['common_ts'].dt.hour

# 删除原common_ts列data.drop(columns=['common_ts'], inplace=True)
data.drop(columns=['common_ts'], inplace=True)

# 将day和time列追加到数据集的最后
data['day'] = data['day']
data['time'] = data['time']

# 保存处理后的数据集到新文件
data.to_csv('../dataset/train_unknown_DayHour.csv', index=False)
