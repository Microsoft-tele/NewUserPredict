# newUserPredict

# Notice: I decide to divide dataset to two parts. One is including udmap, another is not including udmap

## 1. udmap analyse

### There are ten different type of udmap

```json
[
   {"key9":"unknown"},5
   {"key8":"unknown"},3
   {"key7":"unknown"},4
   
   {"key6":"unknown"},
   {"key1":"unknown","key2":"unknown","key6":"unknown"},
   {"key3":"unknown","key1":"unknown","key4":"unknown","key2":"unknown","key5":"unknown"},
   {"key4":"unknown","key5":"unknown"},
   {"key3":"unknown"},
   {"key3":"unknown","key2":"unknown"},
   {"key1":"unknown","key2":"unknown"}
]
```

## 2. Process

1. 首先，你使用Transformer模型将9个key转换为5个嵌入表示的特征。这种嵌入表示可以捕捉特征之间的关系和重要性，有助于将高维的离散特征转换为低维连续向量，以便更好地参与后续的学习任务。

2. 然后，你将这5个嵌入特征与数据集中剩余的10个特征（共15个特征）组合在一起。这样做的目的是将原始特征与嵌入特征结合起来，以丰富模型的输入信息，从而提高模型的性能。

3. 接下来，你将这15个特征输入到一个全连接层中。全连接层可以学习特征之间的复杂关系，并生成更高级别的特征表示。

4. 最后，使用二分类任务来输出两个值，并将其与标准答案进行对比。通过计算损失函数并进行反向传播，使模型逐步调整参数以最小化损失，从而提高模型在二分类任务上的性能。

5.我是小组成员