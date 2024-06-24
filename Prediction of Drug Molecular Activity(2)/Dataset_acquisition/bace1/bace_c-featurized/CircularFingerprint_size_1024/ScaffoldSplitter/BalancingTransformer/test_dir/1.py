import numpy as np
import pandas as pd

# 将分子的标识符保存为 CSV 文件
np.savetxt('shard-0-ids.csv', ids, delimiter=',', fmt='%s')

# 将分子的权重信息保存为 CSV 文件
np.savetxt('shard-0-w.csv', weights, delimiter=',')

# 将分子的特征矩阵保存为 CSV 文件
np.savetxt('shard-0-X.csv', features, delimiter=',')

# 将分子的标签信息保存为 CSV 文件
np.savetxt('shard-0-Y.csv', labels, delimiter=',')

# 加载 CSV 文件
ids = pd.read_csv('shard-0-ids.csv', header=None).values
weights = pd.read_csv('shard-0-w.csv', header=None).values
features = pd.read_csv('shard-0-X.csv', header=None).values
labels = pd.read_csv('shard-0-Y.csv', header=None).values

# 打印数据的维度信息
print('ids shape:', ids.shape)
print('weights shape:', weights.shape)
print('features shape:', features.shape)
print('labels shape:', labels.shape)
