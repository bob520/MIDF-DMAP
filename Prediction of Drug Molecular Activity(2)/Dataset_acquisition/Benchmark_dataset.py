"""
Obtaining benchmark datasets through the Deepchem repository

"""
import deepchem
import pandas as pd

# data_dir: str
#     a directory to save the raw data in
#   save_dir: str
#     a directory to save the dataset in
##The former stores the original data, such as xxx.csv.gz, while the latter stores the processed data, such as returning train_dir, valid_dir, test_dir, tasks.josn

#dc.molnet.featurizers
#featurizer='ECFP'
#featurizer='GraphConv'

#The following two datasets are segmented using Random method
tasks, all_dataset, transformers = deepchem.molnet.load_clintox(
        featurizer='ECFP', splitter='random',
        reload=True,
        save_dir="./clintox1",
        data_dir='./clintox2')

tasks, all_dataset, transformers = deepchem.molnet.load_sider(
        featurizer='ECFP', splitter='random',
        reload=True,
        save_dir="./sider1",
        data_dir='./sider2')


#The following two datasets are segmented using ECFP Scaffold method
tasks, all_dataset, transformers = deepchem.molnet.load_bbbp(
        featurizer='ECFP', splitter='scaffold',
        #read from the local disk (save_dir) or download again when repeating this function
        reload=True,
        save_dir="./bbbp1",
        data_dir='./bbbp2')
#Each segmented dataset has, for example,   train_dataset.ids【SMILES】、train_dataset.y【label】、train_dataset.w【Weight information】、train_dataset.X【Feature matrix】Four variables，
#Usually only ids and y are useful

train_dataset, valid_dataset, test_dataset = all_dataset
print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))
# 1631
# 204
# 204
#all:2039


train_y = train_dataset.y.flatten()
valid_y = valid_dataset.y.flatten()
test_y = test_dataset.y.flatten()


train_df = pd.DataFrame({'smiles': train_dataset.ids, 'label': train_y.astype(int)})
train_df.to_csv("bbbp_train.csv",index=False)

valid_df = pd.DataFrame({'smiles': valid_dataset.ids, 'label': valid_y.astype(int)})
valid_df.to_csv("bbbp_valid.csv",index=False)

test_df = pd.DataFrame({'smiles': test_dataset.ids, 'label': test_y.astype(int)})
test_df.to_csv("bbbp_test.csv",index=False)

train_df['split'] = 'train'
valid_df['split'] = 'valid'
test_df['split'] = 'test'

merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

merged_df.to_excel("../data/BBBP.xlsx", index=False)



tasks, all_dataset, transformers = deepchem.molnet.load_bace_classification(
        featurizer='ECFP', splitter='scaffold',
        reload=True,
        save_dir="./bace1",
        data_dir='./bace2')
train_dataset, valid_dataset, test_dataset = all_dataset

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))
# 1210
# 151
# 152
#all:1513

train_y = train_dataset.y.flatten()
valid_y = valid_dataset.y.flatten()
test_y = test_dataset.y.flatten()

train_df = pd.DataFrame({'smiles': train_dataset.ids, 'label': train_y.astype(int)})
train_df.to_csv("bace_train.csv",index=False)

valid_df = pd.DataFrame({'smiles': valid_dataset.ids, 'label': valid_y.astype(int)})
valid_df.to_csv("bace_valid.csv",index=False)

test_df = pd.DataFrame({'smiles': test_dataset.ids, 'label': test_y.astype(int)})
test_df.to_csv("bace_test.csv",index=False)


train_df['split'] = 'train'
valid_df['split'] = 'valid'
test_df['split'] = 'test'

merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

merged_df.to_excel("../data/bace.xlsx", index=False)

"""
Error message：you may encounter an error
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2050,) + inhomogeneous part.
Problem solving please refer to
https://github.com/deepchem/deepchem/issues/3739

Due to some issues with deepchem itself, errors may occur during local runtime, but it can be successfully executed in Colab
"""