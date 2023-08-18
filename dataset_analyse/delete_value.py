import re

# 读取文本文件
with open('log_analyse', 'r', encoding='utf-16') as file:
    data = file.readlines()

# 使用正则表达式删除指定key[ith]对应的value
pattern = r'("key[1-9]":\s*)"\d+"'
data_modified = [re.sub(pattern, r'\1"unknown"', line) for line in data]

# 将处理后的数据保存为新的文本文件
with open('processed_file.txt', 'w', encoding='utf-8') as file:
    for line in data_modified:
        file.write(line)
