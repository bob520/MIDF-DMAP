# coding=utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import my_dataset, dataset_path

df = pd.read_excel(dataset_path)

if my_dataset == "BACE" or my_dataset == "BBBP" or my_dataset == "BBBPuseScaffold" or my_dataset == "BACEuseScaffold":
    label_counts = df.groupby('split')['label'].value_counts()
    print(label_counts)
else:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    print(label_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='label', y='count', data=label_counts)
    plt.title('Label Counts')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks()
    plt.show()


"""
BBBP dataset
split  label
test   1         161
       0          43
train  1        1244
       0         387
valid  1         155
       0          49
Name: label, dtype: int64
"""