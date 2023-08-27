import pandas as pd

df = pd.read_csv("../dataset/train_processed.csv")
value_counts = df['one_hot'].value_counts()
print(value_counts)