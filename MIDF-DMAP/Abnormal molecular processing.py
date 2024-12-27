"""
Identify the molecular numbers for converting smilies to mol and converting mol to ecfp that RDKit cannot process, and remove them
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from config import dataset_path

df= pd.read_excel(dataset_path)
print("The total size of this dataset is ",len(df))

#List of abnormal molecular indexes
idx_failed_ecfp = []
for idx, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            # An exception occurred while converting smiles to mol
            # When RDKit encounters non-compliant smiles, it cannot convert them to mol form and therefore returns None
            idx_failed_ecfp.append(idx)
        else:
            # Try to see if there is an error when converting to ECFP, and if so, throw an exception
            _ = AllChem.GetMorganFingerprint(mol, radius=1)
    except:
        idx_failed_ecfp.append(idx)

print(f"List of abnormal molecular indexes：{idx_failed_ecfp}")
#Remove these molecules
df = df.drop(df.index[idx_failed_ecfp])
print(len(df))
df.reset_index(inplace=True, drop=True,names="num")
df.to_excel("./data/clintox2.xlsx",index=False)