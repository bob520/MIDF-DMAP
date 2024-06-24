# coding=utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('casp.xlsx')

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
