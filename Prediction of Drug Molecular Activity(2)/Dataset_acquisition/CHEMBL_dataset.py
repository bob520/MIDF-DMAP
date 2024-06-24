import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--receptor', type=str, default='Caspase-1', help='Specify the receptor name')
args = parser.parse_args()


#Create a protein querier instance
target = new_client.target
#Search for a protein
target_query = target.search(args.receptor)
targets = pd.DataFrame.from_dict(target_query)
#Select a target from the returned targets
selected_target = targets.target_chembl_id[1]

activity = new_client.activity
#Obtain all related compounds of protein receptors
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
#Remove null values
compounds_IC50notNULL = df[df.standard_value.notna()]
print("The length of the compound dataset corresponding to this receptor is",len(compounds_IC50notNULL))
compounds_IC50notNULL['standard_value'] = compounds_IC50notNULL['standard_value'].astype(float)

#Using intermediate molecules
# compounds_IC50notNULL['label'] = compounds_IC50notNULL['standard_value'].map(lambda x : 1 if x <= 1000
# else (0 if x >=10000 else 'intermediate'))
# compounds_IC50notNULL = compounds_IC50notNULL[compounds_IC50notNULL['label'] != 'intermediate']

#Not using intermediate molecules
compounds_IC50notNULL['label'] = compounds_IC50notNULL['standard_value'].map(lambda x : 1 if x <= 1000
else 0 )

#Remove excess columns
selection = ['canonical_smiles','label']
df = compounds_IC50notNULL[selection]
df.canonical_smiles.replace('nan',np.nan, inplace=True)
#As long as there is a column with nan, delete that row
df.dropna(inplace=True)
#Rewrite index
df.reset_index(inplace=True,drop=True)
print("The length after cleaning is",len(df))
df = df.rename(columns={'canonical_smiles': 'smiles'})
df.to_excel("../data/compounds_final.xlsx",index=False)
