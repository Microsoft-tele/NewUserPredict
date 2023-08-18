# 读取文本文件
with open('processed_file.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

# 删除每行中的最后6个字符
data_modified = [line[:-7] + '\n' for line in data]

# 将处理后的数据保存为新的文本文件
with open('processed_file_2.txt', 'w', encoding='utf-8') as file:
    for line in data_modified:
        file.write(line)
