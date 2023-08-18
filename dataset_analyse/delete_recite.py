# 读取文本文件
with open('processed_file_3.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

# 使用set来删除重复行
unique_data = [line.rstrip() + "\n" for line in data]

# 将处理后的数据保存为新的文本文件
with open('processed_file_4.txt', 'w', encoding='utf-8') as file:
    for line in unique_data:
        file.write(line)
