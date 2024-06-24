#### Environmental construction

The packages that the project depends on are written in requirements. txt

```
pip install -r requirements.txt
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

#### Project parameter settings

Specify the dataset name, path, and some model hyperparameters in config.py.Both the my_dataset and dataset_path must be specified.


#### Start running the model

```
python main.py
```

#### Abnormal molecular processing

If you encounter an error when executing main.py due to the presence of an exception molecule that RDKit cannot handle, run the following code to remove the exception molecule.

```
python Abnormal molecular processing.py
```

