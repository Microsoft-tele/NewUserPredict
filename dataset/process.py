import pandas as pd
import json

# 读取CSV文件
df = pd.read_csv('train.csv')


# 将JSON格式的'udmap'列转换为字典，处理NaN值和unknown
def convert_udmap(udmap_str):
    try:
        return json.loads(udmap_str)
    except json.JSONDecodeError:
        return {'unknown': True}
    except TypeError:
        return {}


df['udmap'] = df['udmap'].apply(convert_udmap)

# 根据udmap是否为unknown，将数据分成两个DataFrame
df_unknown = df[df['udmap'].apply(lambda x: 'unknown' in x)]
df_known = df[df['udmap'].apply(lambda x: 'unknown' not in x)]

# 保存为两个文件
df_unknown.to_csv('train_unknown.csv', index=False)
df_known.to_csv('train_known.csv', index=False)
