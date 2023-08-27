# NewUserPredict

# Notice: I decide to divide the dataset into two parts. One contains udmap, another dose not contain udmap

## 1. udmap analyse

### There are ten different types of udmap

```json
[
   {"key9":"unknown"},
   {"key8":"unknown"},
   {"key7":"unknown"},
   
   {"key6": "unknown"},
   {"key1":"unknown","key2":"unknown","key6":"unknown"},
   {"key3":"unknown","key1":"unknown","key4":"unknown","key2":"unknown","key5":"unknown"},
   {"key4":"unknown","key5":"unknown"},
   {"key3": "unknown"},
   {"key3":"unknown","key2":"unknown"},
   {"key1":"unknown","key2":"unknown"}
]
```

## 2. Process

1. 首先，你使用Transformer模型将9个key转换为5个嵌入表示的特征。这种嵌入表示可以捕捉特征之间的关系和重要性，有助于将高维的离散特征转换为低维连续向量，以便更好地参与后续的学习任务。

2. 然后，你将这5个嵌入特征与数据集中剩余的10个特征（共15个特征）组合在一起。这样做的目的是将原始特征与嵌入特征结合起来，以丰富模型的输入信息，从而提高模型的性能。

3. 接下来，你将这15个特征输入到一个全连接层中。全连接层可以学习特征之间的复杂关系，并生成更高级别的特征表示。

4. 最后，使用二分类任务来输出两个值，并将其与标准答案进行对比。通过计算损失函数并进行反向传播，使模型逐步调整参数以最小化损失，从而提高模型在二分类任务上的性能。

## 3. Statistic eid

## 以下单独一组
- 合并，unknown使用特征工程补全
1. eid = 26: [17w] {key2, key3}
16. eid = 40 [4k]  {key2, key3}
20. eid = 3  [2k]  {key2, key3}
33. eid = 38 [240] {key2, key3}
14. eid = 25 [5k]  {key2, key3}, {unknown}
12. eid = 12  [6k] {key2, key3}, {unknown}
36. eid = 7  [7]   {key2, key3}, {unknown}

- 合并，多余的key2删除
13. eid = 0   [5k] {key3}
15. eid = 27 [5k]  {key3}
6. eid = 34: [5w]  {key3 all, key2 partial}


## 以下单独一组，没有的补充
- 单独一组
5. eid = 2:  [5w]  {key4, key5}

- 补全key4和key5
7. eid = 5:  [3w]  {key3, key2}, {key1, key2, key3, key4, key5}

## 以下分成一类，都当成unknown
- 合并，多余的key6删除
10. eid = 41: [2w] {key1, key2}
25. eid = 36 [900] {key1, key2}
35. eid = 31 [100] {key1, key2}
22. eid = 30 [1k5] {key1, key2, key6}

- 单独处理
27. eid = 4  [800] {key9}
28. eid = 1  [736] {key9}
29. eid = 19 [700] {key8}
30. eid = 13 [700] {key7}
18. eid = 15 [3k]  {key6}
23. eid = 20 [1k5] {unknown}
24. eid = 10 [1k]  {unknown}
21. eid = 9  [1k5] {unknown}
19. eid = 29 [2k]  {unknown}
17. eid = 37 [3k5] {unknown}
11. eid = 32: [6k] {unknown}
8. eid = 21: [3w]  {unknown}
9. eid = 39: [2w]  {unknown}
2. eid = 35: [8w]  {unknown}
3. eid = 11: [5w]  {unknown}
4. eid = 8:  [5w]  {unknown}
26. eid = 33 [800] {unknown}
31. eid = 42 [528] {unknown}
32. eid = 28 [300] {unknown}
34. eid = 14 [210] {unknown}
37. eid = 16 [6]   {unknown}
38. eid = 23 [3]   {unknown}
39. eid = 6  [2]   {unknown}
40. eid = 22, 18, 17, 24 [1]   {unknown}