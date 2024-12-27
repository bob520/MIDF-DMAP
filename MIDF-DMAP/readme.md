#### Environmental construction

The packages that the project depends on are written in requirements. txt
Please execute the following command in the current project path

```
conda create --name DMAP python=3.9
conda activate DMAP
pip install -r requirements.txt
```
Due to the influence of cuda versions on the installation of the torch library, different computers may require different versions. Therefore, this project did not specify torch=2.1.0+cu121 in requirements. txt. In order to facilitate you in downloading the torch version you need, we will list this operation separately.

Please download the required WHL file from this website to this project.
For example, downloadingcu121/torch-2.1.0%2Bcu121-cp39-cp39-win_amd64.whl.

```
https://download.pytorch.org/whl/torch_stable.html
```
Then execute the following instructions
```
pip install .\torch-2.1.0+cu121-cp39-cp39-win_amd64.whl
```

**There are two points to note when building the environment**

**(1)** The PGL library requires version 2.2.5, otherwise an error message will be generated: ModuleNotFoundError: No module named 'pad. fluid'. However, the maximum version of the pgl library on pip is 2.2.3, and installing directly on pip will result in errors. And the GitHub version of this library is 2.2.5, so we chose to install it through GitHub.

```
git clone https://github.com/PaddlePaddle/PGL.git
```

After decompression, enter this folder

```
python setup.py install
```

**(2)** Due to the earlier release time of mol2vec and its incompatibility with Gensim versions 4.0.0 and above, errors may occur during runtime

```
for x in sentences2vec(df['ecfp'], model, unseen='UNK'):
  File "C:\Users\ThinkStation2\Anaconda\envs\DMAP\lib\site-packages\mol2vec\features.py", line 425, in sentences2vec
    keys = set(model.wv.vocab.keys())
AttributeError: The vocab attribute was removed from KeyedVector in Gensim 4.0.0
```

Please redirect to the error reporting line. The original code of this line is

```
keys = set(model.wv.vocab.keys())
```

Modify it to

```
keys = set(model.wv.index_to_key)
```

#### Dataset collection

Please run the code in the Dataset_acquisition folder to collect the compound molecular dataset for model training.

If you need to use a benchmark dataset, please run:

```
python Dataset_acquisition/Benchmark_dataset.py
```

If you need to collect a dataset from the CHEMBL database, please run:

```
python Dataset_acquisition/CHEMBL_dataset.py --receptor receptorname
```

Next, divide the datasets that require the use of the scafold segmentation method, please run:

```
python Dataset_acquisition/Scaffold_split_dataset.py
```


#### Project parameter settings

Specify the dataset name, path, and some model hyperparameters in config.py. Both the my_dataset and dataset_path must be specified.

#### Start running the model

Running the following code will train the model and store the best performing model on the validation set.

```
python main.py
```

#### Test model performance

Read the best stored model and test its performance on the testset.

```
python test_code.py
```

#### Abnormal molecular processing

If you encounter an error when executing main.py due to the presence of an exception molecule that RDKit cannot handle, run the following code to remove the exception molecule.

```
python Abnormal molecular processing.py
```
#### Obtain the distribution of dataset samples
```
python sample distribution.py
```