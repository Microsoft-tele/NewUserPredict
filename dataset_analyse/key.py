import pandas as pd
import json
import sys

# 读取CSV文件
df = pd.read_csv('./train.csv')


# 将JSON格式的'udmap'列转换为字典，处理NaN值
def convert_udmap(udmap_str):
    try:
        return json.loads(udmap_str)
    except json.JSONDecodeError:
        return {'unknown': True}
    except TypeError:  # 处理NaN值的情况
        return {}


df['udmap'] = df['udmap'].apply(convert_udmap)

if __name__ == '__main__':
    for i in range(1, 10):
        # 用于统计'key ith'特征的值的字典
        keyi_values_count = {}

        # 遍历每个字典，提取'keyi'的值并统计
        key = 'key' + str(i)
        for udmap_dict in df['udmap']:
            if key in udmap_dict:
                keyi_value = udmap_dict[key]
                keyi_values_count[keyi_value] = keyi_values_count.get(keyi_value, 0) + 1

        # 按照字典的值（数量）进行降序排序
        sorted_keyi_values_count = dict(sorted(keyi_values_count.items(), key=lambda item: item[1], reverse=True))

        # 将输出重定向到文件
        with open(f'output_key{i}.json', 'w', encoding='utf8') as f:
            sys.stdout = f

            # 输出'keyi'特征值的计数（按数量降序排列）
            print(f"'key{i}'特征值的计数（按数量降序排列）：")
            print(json.dumps(sorted_keyi_values_count, indent=4))

        # 恢复标准输出
        sys.stdout = sys.__stdout__
